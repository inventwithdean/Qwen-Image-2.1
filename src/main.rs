use std::{path::Path, time::Instant};

use burn::{
    Tensor,
    backend::wgpu::WgpuDevice,
    module::{Module, Quantizer},
    store::ModuleRecord,
    tensor::{
        Device, Distribution, FloatDType, Int,
        quantization::{Calibration, QuantScheme, ScaleDtype},
        s,
    },
};
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use qwen_image::{
    qwen_image::{QwenImageBlockStreamer, QwenImageTransformerModelConfig},
    qwen_image_modules::{KvCacheMode, QwenImageKVCache, QwenImageTransformerBlockConfig},
};

fn read_f32(path: &str) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
        .collect()
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

fn main() {
    let device: Device = WgpuDevice::default().into();

    let config = QwenImageTransformerModelConfig::new([16, 56, 56]);

    let record = ModuleRecord::load("out/shell.mpk").expect("shell");
    let model = config.init(&device).load_record(record);
    let mut streamer = QwenImageBlockStreamer::new("out", &config, &device);

    let prompt_floats: Vec<f32> = read_f32("prompt_embeds.bin");
    let t_text = prompt_floats.len() / 4096;
    let (h, w) = (32_usize, 32_usize);
    let target_tokens = h * w;
    let slots = target_tokens / 4;
    let img_shapes = vec![(1, h, w)];
    let encoder_hidden_states =
        Tensor::<1>::from_floats(prompt_floats.as_slice(), &device).reshape([1, t_text, 4096]);

    let mut latents = Tensor::<3>::random(
        [1, target_tokens, 64],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    let mask_ints: Vec<i64> = (0..t_text + slots).map(|i| (i >= t_text) as i64).collect();
    let img_mask = Tensor::<1, Int>::from_ints(mask_ints.as_slice(), &device)
        .reshape([1, t_text + slots])
        .bool();

    let steps = 25;
    let sigmas: Vec<f32> = schedule(steps, target_tokens);

    let mut kv_cache = QwenImageKVCache::new(streamer.num_layers, &device);

    let total = Instant::now();
    for step in 0..steps {
        let t = Instant::now();

        let mode = if step == 0 {
            KvCacheMode::EXTRACT
        } else {
            KvCacheMode::CACHED
        };
        let timestep = Tensor::<1>::from_floats([sigmas[step]], &device);

        let out = model.forward(
            &mut streamer,
            latents.clone(),
            encoder_hidden_states.clone(),
            timestep,
            img_shapes.clone(),
            img_mask.clone(),
            None,
            Some(&mut kv_cache),
            Some(mode),
        );

        let n = out.dims()[1];
        let pred = out.slice(s![.., n - target_tokens..n, ..]);

        // Euler step
        latents = latents + pred * (sigmas[step + 1] - sigmas[step]);

        // Readback forces a sync so the timing is real
        let _ = latents.clone().sum_dim(0).sum_dim(1).sum_dim(2).into_data();
        let sample = latents
            .clone()
            .slice(s![0..1, 0..1, 0..5])
            .cast(FloatDType::F32)
            .into_data()
            .try_into_vec::<f32>()
            .unwrap();
        println!("Step {:>2} sample: {:?}", step, sample);

        let step_time = t.elapsed();
        println!("step {step:>2}: {step_time:.2?}",);
    }
    println!("{steps} steps: {:.2?}", total.elapsed());

    let final_data = latents
        .cast(FloatDType::F32)
        .into_data()
        .try_into_vec::<f32>()
        .unwrap();

    let out_bytes: Vec<u8> = final_data
        .into_iter()
        .flat_map(|f| f.to_ne_bytes())
        .collect();

    std::fs::write("latents_out.bin", out_bytes).expect("Failed to save latents_out.bin");
    println!("Saved latents_out.bin.");
}

// Converts from F32 to INT8 blocks for ram/disk offloading
fn _convert_qwen() {
    const SHARDS: [&str; 2] = [
        "diffusion_pytorch_model-00001-of-00002.safetensors",
        "diffusion_pytorch_model-00002-of-00002.safetensors",
    ];
    const NUM_LAYERS: usize = 32;
    let src = "./weights_f32";
    let out = Path::new("out");
    std::fs::create_dir_all(out).unwrap();
    let device: Device = WgpuDevice::default().into();
    let mut quantizer = Quantizer::new(
        Calibration::MinMax,
        QuantScheme::default().per_tensor(ScaleDtype::F32),
    );

    {
        let mut shell = QwenImageTransformerModelConfig::new([16, 56, 56])
            .with_num_layers(0)
            .init(&device);

        for shard in SHARDS {
            let mut store = SafetensorsStore::from_file(Path::new(&src).join(shard))
                .with_from_adapter(PyTorchToBurnAdapter)
                .allow_partial(true);
            let r = shell.load_from(&mut store).unwrap();
            println!(
                "shell <- {shard}: applied {}, errors {:?}",
                r.applied.len(),
                r.errors
            );
        }
        shell
            .into_record()
            .save(out.join("shell.mpk"))
            .expect("save shell");
    }

    for i in 0..NUM_LAYERS {
        let mut block = QwenImageTransformerBlockConfig::new(4096, 32, 128, 3, 1e-6).init(&device);

        for shard in SHARDS {
            let mut store = SafetensorsStore::from_file(Path::new(&src).join(shard))
                .with_from_adapter(PyTorchToBurnAdapter)
                .with_key_remapping(&format!(r"^transformer_blocks\.{i}\."), "")
                .allow_partial(true);
            let r = block.load_from(&mut store).unwrap();
            println!(
                "block {i} <- {shard}: applied {}, errors {:?}",
                r.applied.len(),
                r.errors
            );
        }

        let block = block.quantize_weights(&mut quantizer);
        block
            .into_record()
            .save(out.join(format!("block_{i}.mpk")))
            .expect("save block");
        println!("saved block_{i}.mpk");
    }
}
