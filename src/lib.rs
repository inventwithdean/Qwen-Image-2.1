pub mod normalization;
pub mod qwen_image;
pub mod qwen_image_modules;
pub mod vae;

use burn::{
    module::Module,
    store::ModuleRecord,
    tensor::{Bytes, Device, Distribution, FloatDType, Tensor, s},
};
use qwen_image::{
    QwenImageBlockStreamerWeb, QwenImageTransformerModel, QwenImageTransformerModelConfig,
    SyncHandle,
};
use qwen_image_modules::{KvCacheMode, QwenImageKVCache};
use wasm_bindgen::prelude::*;

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
    device: Option<Device>,
}

#[wasm_bindgen]
impl QwenWeb {
    #[wasm_bindgen(constructor)]
    pub fn new() -> QwenWeb {
        console_error_panic_hook::set_once();
        QwenWeb {
            model: None,
            streamer: None,
            device: None,
        }
    }

    /// Init WebGPU, build the layer-less model shell, and prepare the streamer.
    pub async fn load_shell(&mut self, shell_bytes: &[u8]) -> Result<(), JsValue> {
        log("1. Starting async load. Requesting WebGPU Adapter...");

        let device = Device::wgpu_options()
            .init_async()
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        log("2. WebGPU Initialized. Parsing shell record...");
        let config = QwenImageTransformerModelConfig::new([16, 56, 56]);
        let mut model = config.clone().with_num_layers(0).init(&device);

        let shell_record = ModuleRecord::from_bytes(Bytes::from_bytes_vec(shell_bytes.to_vec()))
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        model = model.load_record(shell_record);

        let streamer = QwenImageBlockStreamerWeb::new_empty(&config, &device);

        self.model = Some(model);
        self.streamer = Some(streamer);
        self.device = Some(device);
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

    pub async fn probe_f16(&self) -> Result<String, JsValue> {
        let device = self.device.as_ref().ok_or("no device")?;
        let a = Tensor::<2>::random([256, 16384], Distribution::Normal(0.0, 1.0), device);
        let b = Tensor::<2>::random([16384, 256], Distribution::Normal(0.0, 1.0), device);
        let r32 = a.clone().matmul(b.clone());
        let r16 = a
            .cast(FloatDType::F16)
            .matmul(b.cast(FloatDType::F16))
            .cast(FloatDType::F32);
        let err = (r16 - r32.clone()).abs().max() / r32.abs().max();
        let v = err
            .into_data_async()
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?
            .try_into_vec::<f32>()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(format!("max relative error: {:e}", v[0]))
    }

    pub async fn generate(
        &mut self,
        prompt_floats: &[f32],
    ) -> Result<js_sys::Float32Array, JsValue> {
        let res = self.probe_f16().await?;
        log(&res);
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
        let (h, w) = (32_usize, 32_usize);
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

        let steps = 25;
        let sigmas: Vec<f32> = schedule(steps, target_tokens);

        let mut kv_cache = QwenImageKVCache::new(streamer.num_layers, batch_size, device);

        for step in 0..steps {
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

        let final_data = latents
            .cast(FloatDType::F32)
            .into_data_async()
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?
            .try_into_vec::<f32>()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(js_sys::Float32Array::from(final_data.as_slice()))
    }
}
