pub mod normalization;
pub mod qwen_image;
pub mod qwen_image_modules;
pub mod vae;

use burn::{
    module::Module,
    store::ModuleRecord,
    tensor::{Bytes, Device, Distribution, FloatDType, Tensor, TensorData, s},
};
use qwen_image::{
    QwenImageBlockStreamerWeb, QwenImageTransformerModel, QwenImageTransformerModelConfig,
    SyncHandle,
};
use qwen_image_modules::{KvCacheMode, QwenImageKVCache};
use wasm_bindgen::prelude::*;

use crate::vae::model::{AutoencoderKLQwenImage, AutoencoderKLQwenImageConfig};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

fn schedule(steps: usize, seq_len: usize) -> Vec<f32> {
    let (base_shift, max_shift) = (0.5f64, 0.9f64);
    let (base_len, max_len) = (256.0f64, 8192.0f64);
    let terminal = 0.02f64;

    let m = (max_shift - base_shift) / (max_len - base_len);
    let mu = seq_len as f64 * m + (base_shift - m * base_len);
    let emu = mu.exp();

    let mut s: Vec<f64> = (0..steps)
        .map(|i| {
            let x = 1.0 + (1.0 / steps as f64 - 1.0) * i as f64 / (steps - 1) as f64;
            emu / (emu + (1.0 / x - 1.0))
        })
        .collect();

    let scale = (1.0 - s[steps - 1]) / (1.0 - terminal);
    for v in s.iter_mut() {
        *v = 1.0 - (1.0 - *v) / scale;
    }
    s.push(0.0);
    s.into_iter().map(|v| v as f32).collect()
}

#[wasm_bindgen]
pub struct QwenWeb {
    model: Option<QwenImageTransformerModel>,
    streamer: Option<QwenImageBlockStreamerWeb>,
    vae: Option<AutoencoderKLQwenImage>,
    device: Option<Device>,
}

pub struct LatentNorm {
    mean: Tensor<4>, // (1, 64, 1, 1)
    std: Tensor<4>,
}

impl LatentNorm {
    pub fn new(mean: &[f32], std: &[f32], device: &Device) -> Self {
        let make = |v: &[f32]| {
            Tensor::<4>::from_data(TensorData::new(v.to_vec(), [1, v.len(), 1, 1]), device)
        };
        Self {
            mean: make(mean),
            std: make(std),
        }
    }

    /// diffusion-space latents -> VAE-space (use before decode)
    pub fn denormalize(&self, z: Tensor<4>) -> Tensor<4> {
        z * self.std.clone() + self.mean.clone()
    }

    /// VAE-space latents -> diffusion-space (use after encode)
    pub fn normalize(&self, z: Tensor<4>) -> Tensor<4> {
        (z - self.mean.clone()) / self.std.clone()
    }
}

#[wasm_bindgen]
pub struct DecodedImage {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

#[wasm_bindgen]
impl DecodedImage {
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }
    /// Uint8ClampedArray, ready for `new ImageData(...)`
    #[wasm_bindgen(getter)]
    pub fn rgba(&self) -> js_sys::Uint8ClampedArray {
        js_sys::Uint8ClampedArray::from(self.rgba.as_slice())
    }
}

impl QwenWeb {}

#[wasm_bindgen]
impl QwenWeb {
    #[wasm_bindgen(constructor)]
    pub fn new() -> QwenWeb {
        console_error_panic_hook::set_once();
        QwenWeb {
            model: None,
            streamer: None,
            vae: None,
            device: None,
        }
    }

    fn load_vae(&mut self, vae_bytes: Vec<u8>) -> Result<(), JsValue> {
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Call load_shell first!"))?;
        let record = ModuleRecord::from_bytes(Bytes::from_bytes_vec(vae_bytes))
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        self.vae = Some(
            AutoencoderKLQwenImageConfig::new()
                .init(device)
                .load_record(record),
        );
        Ok(())
    }

    pub async fn init_device(&mut self) -> Result<(), JsValue> {
        log("Initializing Device...");
        let device = Device::wgpu_options()
            .init_async()
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        self.device = Some(device);
        log("Device initialized successfully!");
        Ok(())
    }

    /// Init WebGPU, build the layer-less model shell, and prepare the streamer.
    pub async fn load_shell(&mut self, shell_bytes: &[u8]) -> Result<(), JsValue> {
        log("Loading DiT shell...");
        let device = self.device.as_ref().expect("Call init_device first!");
        let config = QwenImageTransformerModelConfig::new([16, 56, 56]);
        let mut model = config.clone().with_num_layers(0).init(&device);

        let shell_record = ModuleRecord::from_bytes(Bytes::from_bytes_vec(shell_bytes.to_vec()))
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        model = model.load_record(shell_record);

        log("Preparing Transformer Block Streamer...");
        let streamer = QwenImageBlockStreamerWeb::new_empty(&config, &device);

        self.model = Some(model);
        self.streamer = Some(streamer);
        Ok(())
    }

    pub fn num_layers(&self) -> Result<usize, JsValue> {
        self.streamer
            .as_ref()
            .map(|s| s.num_layers)
            .ok_or_else(|| JsValue::from_str("call load_shell first"))
    }

    /// Register the OPFS FileSystemSyncAccessHandle for the next layer, in order.
    pub fn add_block_handle(&mut self, handle: SyncHandle) -> Result<(), JsValue> {
        let streamer = self
            .streamer
            .as_mut()
            .ok_or_else(|| JsValue::from_str("call load_shell first"))?;
        let idx = streamer
            .add_handle(handle)
            .map_err(|e| JsValue::from_str(&e))?;
        log(&format!(
            "   - Registered block {} ({:.1} MiB on disk)",
            idx,
            streamer.block_size(idx) as f64 / (1024.0 * 1024.0)
        ));
        Ok(())
    }

    /// Release the OPFS sync handles (call before dropping the worker / clearing cache).
    pub fn close_blocks(&mut self) {
        if let Some(s) = self.streamer.as_mut() {
            s.close_all();
        }
    }

    pub fn release_vae(&mut self) {
        self.vae = None;
        self.device
            .as_ref()
            .expect("No device found!")
            .memory_cleanup();
    }

    pub fn release_dit(&mut self) {
        self.close_blocks();
        self.model = None;
        self.streamer = None;
        self.device
            .as_ref()
            .expect("No device found!")
            .memory_cleanup();
    }
}

#[wasm_bindgen]
impl QwenWeb {
    pub async fn generate(
        &mut self,
        prompt_floats: &[f32],
        h: usize,
        w: usize,
        num_steps: usize, // Number of steps
    ) -> Result<js_sys::Float32Array, JsValue> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| JsValue::from_str("call load_shell first"))?;
        let streamer = self.streamer.as_mut().unwrap();
        let device = self.device.as_ref().unwrap();

        if !streamer.is_ready() {
            return Err(JsValue::from_str("not all block handles are registered"));
        }

        let t_text = prompt_floats.len() / 4096;
        let batch_size = 1;
        let target_tokens = h * w;
        let slots = target_tokens / 4;
        let img_shapes = vec![(1, h, w)];

        let encoder_hidden_states =
            Tensor::<1>::from_floats(prompt_floats, device).reshape([1, t_text, 4096]);
        let encoder_hidden_states = encoder_hidden_states.repeat_dim(0, batch_size);

        let mut latents = Tensor::<3>::random(
            [batch_size, target_tokens, 64],
            Distribution::Normal(0.0, 1.0),
            device,
        )
        .cast(FloatDType::F32);

        let sigmas: Vec<f32> = schedule(num_steps, target_tokens);

        let mut kv_cache = QwenImageKVCache::new(streamer.num_layers, batch_size, device);

        for step in 0..num_steps {
            let mode = if step == 0 {
                KvCacheMode::EXTRACT
            } else {
                KvCacheMode::CACHED
            };
            let timestep = Tensor::<1>::from_floats([sigmas[step]], device);
            // was: Tensor<1, Int>::from_ints(...).reshape(...).bool().repeat_dim(...)
            let img_mask: Vec<bool> = (0..t_text + slots).map(|i| i >= t_text).collect();

            let out = model.forward(
                streamer,
                latents.clone(),
                encoder_hidden_states.clone(),
                timestep,
                img_shapes.clone(),
                &img_mask,
                None,
                Some(&mut kv_cache),
                Some(mode),
            );

            let n = out.dims()[1];
            let pred = out
                .slice(s![.., n - target_tokens..n, ..])
                .cast(FloatDType::F32);
            latents = latents + pred * (sigmas[step + 1] - sigmas[step]);

            // Yield to the event loop + drain the WebGPU queue each step.
            let _ = latents
                .clone()
                .sum_dim(0)
                .sum_dim(1)
                .sum_dim(2)
                .into_data_async()
                .await;

            log(&format!("Completed step {}", step + 1));
        }

        let data = latents
            .cast(FloatDType::F32)
            .into_data_async()
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?
            .try_into_vec::<f32>()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(js_sys::Float32Array::from(data.as_slice()))
    }

    pub async fn decode_latents(
        &mut self,
        latents: &[f32],    // (1, h*w, 64) straight from generate
        vae_bytes: Vec<u8>, // ignored if a VAE is already loaded via load_vae
        h: usize,           // latent height
        w: usize,           // latent width
    ) -> Result<DecodedImage, JsValue> {
        log("Loading VAE");
        if self.vae.is_none() {
            self.load_vae(vae_bytes)?;
        }
        log("Loaded VAE!");
        let vae = self.vae.as_ref().expect("VAE Not loaded!");
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| JsValue::from_str("call load_shell first"))?;

        // (1, h*w, 64) -> (1, h, w, 64) -> (1, 64, h, w), then z * std + mean
        let z = Tensor::<1>::from_floats(latents, device)
            .reshape([1, h, w, 64])
            .permute([0, 3, 1, 2]);
        let z = LatentNorm::new(&vae.mean, &vae.std, device).denormalize(z);

        let out = self.vae.as_ref().expect("VAE not loaded!").decode(z);

        let [_, _, oh, ow] = out.dims();
        let data = out
            .into_data_async()
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?
            .try_into_vec::<f32>()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let mut rgba = vec![0u8; oh * ow * 4];
        for y in 0..oh {
            for x in 0..ow {
                for c in 0..4 {
                    let v = data[c * oh * ow + y * ow + x];
                    rgba[(y * ow + x) * 4 + c] =
                        ((v + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        Ok(DecodedImage {
            width: ow as u32,
            height: oh as u32,
            rgba,
        })
    }
}
