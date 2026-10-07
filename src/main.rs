use std::{path::Path, time::Instant};

use burn::{
    Tensor,
    module::{Module, Quantizer},
    store::ModuleRecord,
    tensor::{
        Device, Distribution, FloatDType, Int,
        quantization::{Calibration, QuantScheme, ScaleDtype},
        s,
    },
};
use burn_store::{
    BurnpackStore, FloatCastAdapter, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter,
    SafetensorsStore,
};
use qwen_image::{
    qwen_image::{
        QwenImageBlockStreamer, QwenImageTransformerModel, QwenImageTransformerModelConfig,
    },
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
    let device: Device = Device::wgpu(Default::default());

    // Convert BF16 weights to int8 .bpk files
    // _convert_to_bpk(&device);
    let config = QwenImageTransformerModelConfig::new([16, 56, 56]);

    let src = "./out";
    // Don't load layers, we'll stream them from RAM
    let mut model = config.clone().with_num_layers(0).init(&device);
    let shell_record = ModuleRecord::load(Path::new(src).join("shell.bpk")).unwrap();
    model = model.load_record(shell_record);

    let mut streamer = QwenImageBlockStreamer::new(src, &config, &device);

    let file_name = format!("prompt_embeds.bin");
    let latents = generate_image(&model, &mut streamer, &device, &file_name);
    let final_data = latents
        .cast(FloatDType::F32)
        .into_data()
        .try_into_vec::<f32>()
        .unwrap();

    let out_bytes: Vec<u8> = final_data
        .into_iter()
        .flat_map(|f| f.to_ne_bytes())
        .collect();
    std::fs::write(format!("latents_out.bin"), out_bytes).expect("Failed to save latents_out.bin!");
    println!("Saved latents_out.bin!");
}

/// Returns latent vectors.
fn generate_image(
    model: &QwenImageTransformerModel,
    streamer: &mut QwenImageBlockStreamer,
    device: &Device,
    file_name: &str,
) -> Tensor<3> {
    let prompt_floats: Vec<f32> = read_f32(file_name);
    let t_text = prompt_floats.len() / 4096;

    // Generation Config
    let (h, w) = (48_usize, 48_usize);
    let batch_size = 1;

    let target_tokens = h * w;
    let slots = target_tokens / 4;
    let img_shapes = vec![(1, h, w)];
    let encoder_hidden_states =
        Tensor::<1>::from_floats(prompt_floats.as_slice(), device).reshape([1, t_text, 4096]);

    let encoder_hidden_states = encoder_hidden_states.repeat_dim(0, batch_size);

    let mut latents = Tensor::<3>::random(
        [batch_size, target_tokens, 64],
        Distribution::Normal(0.0, 1.0),
        device,
    )
    .cast(FloatDType::F32);

    let mask_ints: Vec<i64> = (0..t_text + slots).map(|i| (i >= t_text) as i64).collect();
    let img_mask = Tensor::<1, Int>::from_ints(mask_ints.as_slice(), device)
        .reshape([1, t_text + slots])
        .bool();
    let img_mask = img_mask.repeat_dim(0, batch_size);

    let steps = 25;
    let sigmas: Vec<f32> = schedule(steps, target_tokens);

    let mut kv_cache = QwenImageKVCache::new(streamer.num_layers, batch_size, device);

    let total = Instant::now();
    for step in 0..steps {
        let t = Instant::now();

        let mode = if step == 0 {
            KvCacheMode::EXTRACT
        } else {
            KvCacheMode::CACHED
        };
        let timestep = Tensor::<1>::from_floats([sigmas[step]], device);

        let out = model.forward(
            streamer,
            latents.clone(), //.cast(FloatDType::BF16),
            encoder_hidden_states.clone(),
            timestep,
            img_shapes.clone(),
            img_mask.clone(),
            None,
            Some(&mut kv_cache),
            Some(mode),
        );

        let n = out.dims()[1];
        let pred = out
            .slice(s![.., n - target_tokens..n, ..])
            .cast(FloatDType::F32);

        // Euler step
        latents = latents + pred * (sigmas[step + 1] - sigmas[step]);

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
    latents
}

/// Converts BF16 safetensors to burn .mpk files with transformer blocks quantized to int8
fn _convert_to_bpk(device: &Device) {
    const SHARDS: [&str; 2] = [
        "diffusion_pytorch_model-00001-of-00002.safetensors",
        "diffusion_pytorch_model-00002-of-00002.safetensors",
    ];
    const NUM_LAYERS: usize = 32;

    let src = "./weights/transformer";
    let out = Path::new("out");
    std::fs::create_dir_all(out).unwrap();

    let from_adapter = PyTorchToBurnAdapter.chain(FloatCastAdapter::to(FloatDType::F32.into()));

    let mut shell = QwenImageTransformerModelConfig::new([16, 56, 56])
        .with_num_layers(0)
        .init(&device);

    for shard in SHARDS {
        let mut store = SafetensorsStore::from_file(Path::new(src).join(shard))
            .with_from_adapter(from_adapter.clone())
            .allow_partial(true);
        shell.load_from(&mut store).unwrap();
    }

    // The shell isn't being quantized and is stored in F32.
    // let quant_scheme = QuantScheme::default().with_value(QuantValue::Q4F);
    // let mut quantizer = Quantizer::new(Calibration::MinMax, quant_scheme);
    // shell = shell.quantize_weights(&mut quantizer);

    let mut out_store = BurnpackStore::from_file(out.join("shell.bpk"));
    shell.save_into(&mut out_store).unwrap();
    println!("saved out/shell.bpk");

    let inner_dim = 32 * 128;
    let block_config = QwenImageTransformerBlockConfig::new(inner_dim, 32, 128, 3, 1e-6);

    let quant_scheme = QuantScheme::default().per_block([64], ScaleDtype::F32);
    let mut quantizer = Quantizer::new(Calibration::MinMax, quant_scheme);

    for i in 0..NUM_LAYERS {
        let mut block = block_config.init(&device);
        for shard in SHARDS {
            let mut store = SafetensorsStore::from_file(Path::new(src).join(shard))
                .with_from_adapter(from_adapter.clone())
                .with_key_remapping(&format!(r"^transformer_blocks\.{i}\."), "")
                .allow_partial(true);
            block.load_from(&mut store).unwrap();
        }

        let block = block.quantize_weights(&mut quantizer);

        let mut out_store = BurnpackStore::from_file(out.join(format!("block_{i}.bpk")));
        block.save_into(&mut out_store).unwrap();
        println!("saved out/block_{i}.bpk");
    }
}
