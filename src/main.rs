use burn::{
    module::{Module, Quantizer},
    tensor::{
        DType, Device,
        quantization::{Calibration, QuantScheme, ScaleDtype},
    },
};
use burn_store::{
    FloatCastAdapter, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
};
use qwen_image::text_encoder::{
    model::Qwen3VLTextModelConfig, text_encoder_modules::Qwen3VLTextDecoderLayerConfig,
};

fn main() {}

fn _convert_text_encoder() {
    let device = Device::wgpu(Default::default());
    let mut text_encoder = Qwen3VLTextModelConfig::new()
        .with_num_hidden_layers(0)
        .init(&device);

    let path = "C:/Users/inven/.cache/huggingface/hub/models--Qwen--Qwen-Image-2.1/snapshots/d26bb61231c349cf6b7896fa83353113880e1ba3/text_encoder";
    let shards = [
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
    ];
    let adapter = PyTorchToBurnAdapter.chain(FloatCastAdapter::to(DType::F32));

    for shard in shards {
        let mut store = SafetensorsStore::from_file(format!("{}/{}", path, shard))
            .allow_partial(true)
            .with_from_adapter(adapter.clone())
            .with_key_remapping(r"^model\.language_model\.", "");
        let result = text_encoder.load_from(&mut store).unwrap();
        if result.applied.len() > 0 {
            println!("Applied: {:?}", result.applied);
        }
    }

    println!("Loaded Shell!");

    let scheme = QuantScheme::default().per_block([64], ScaleDtype::F32);
    let mut quantizer = Quantizer::new(Calibration::MinMax, scheme);

    let text_encoder = text_encoder.quantize_weights(&mut quantizer);
    text_encoder
        .save_file("out/text_encoder/shell.bpk")
        .unwrap();

    let num_hidden_layers = 36;
    let mut block = Qwen3VLTextDecoderLayerConfig::new(4096, 128, 32, 8, 12288, 1e-6).init(&device);
    for idx in 0..num_hidden_layers {
        println!("Loading block {idx}");
        for shard in shards {
            let mut store = SafetensorsStore::from_file(format!("{}/{}", path, shard))
                .allow_partial(true)
                .with_from_adapter(adapter.clone())
                .with_key_remapping(format!(r"^model\.language_model\.layers\.{idx}\."), ""); // Make sure only this layer's weights are there

            let result = block.load_from(&mut store).unwrap();
            if result.applied.len() > 0 {
                println!("Applied: {:?}", result.applied);
            }
        }
        let quantized_block = block.clone().quantize_weights(&mut quantizer);
        // After the block is loaded with correct weights
        quantized_block
            .save_file(format!("out/text_encoder/block_{idx}.bpk"))
            .unwrap();
    }
}
