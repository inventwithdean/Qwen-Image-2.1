use std::time::Instant;

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
use qwen_image::vae::model::AutoencoderKLQwenImageConfig;

use burn::tensor::{Tensor, TensorData};
use image::{RgbaImage, imageops::FilterType};

const MAX_SIDE: u32 = 512; // start small; raise once it works

fn load_image(path: &str) -> (RgbaImage, u32, u32) {
    let mut img = image::open(path).unwrap().to_rgba8(); // alpha = 255 if the source has none
    let (w, h) = img.dimensions();

    // Downscale if large, then crop to a multiple of 16 (the VAE's compression ratio).
    if w.max(h) > MAX_SIDE {
        let s = MAX_SIDE as f32 / w.max(h) as f32;
        img = image::imageops::resize(
            &img,
            (w as f32 * s) as u32,
            (h as f32 * s) as u32,
            FilterType::Lanczos3,
        );
    }
    let (w, h) = (img.width() / 16 * 16, img.height() / 16 * 16);
    let img = image::imageops::crop_imm(&img, 0, 0, w, h).to_image();
    (img, w, h)
}

/// RGBA8 (HWC, interleaved) -> (1, 4, H, W) f32 in [-1, 1]
fn to_tensor(img: &RgbaImage, device: &Device) -> Tensor<4> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut data = vec![0f32; 4 * h * w];
    for (x, y, px) in img.enumerate_pixels() {
        for c in 0..4 {
            data[c * h * w + y as usize * w + x as usize] = px[c] as f32 / 127.5 - 1.0;
        }
    }
    Tensor::from_data(TensorData::new(data, [1, 4, h, w]), device)
}

/// (1, 4, H, W) in [-1, 1] -> RGBA8
fn to_image(t: Tensor<4>) -> RgbaImage {
    let [_, _, h, w] = t.dims();
    let data = t.into_data().try_to_vec::<f32>().unwrap();
    let mut img = RgbaImage::new(w as u32, h as u32);
    for (x, y, px) in img.enumerate_pixels_mut() {
        for c in 0..4 {
            let v = data[c * h * w + y as usize * w + x as usize];
            px[c] = ((v + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8;
        }
    }
    img
}

fn main() {
    let device = Device::wgpu(Default::default());
    let mut vae = AutoencoderKLQwenImageConfig::new().init(&device);
    let vae_path = "./weights/vae/diffusion_pytorch_model.safetensors";
    let adapter = PyTorchToBurnAdapter.chain(FloatCastAdapter::to(DType::F32));
    let mut store = SafetensorsStore::from_file(vae_path).with_from_adapter(adapter);
    let result = vae.load_from(&mut store).unwrap();
    println!("{}", result);
    vae.clone().save_file("./out/vae.bpk").unwrap();

    let (img, w, h) = load_image("latents_out.png");
    let x = to_tensor(&img, &device);

    let t = Instant::now();
    let (mean, _) = vae.encode(x.clone());
    let _ = mean.clone().into_data(); // forces completion
    println!("encode: {:?}", t.elapsed());
    println!("latent shape: {:?}", mean.dims()); // expect [1, 64, h/16, w/16]

    let t = Instant::now();
    let recon = vae.decode(mean);
    let _ = recon.clone().into_data();
    println!("recon shape: {:?}", recon.dims()); // expect [1, 4, h, w]

    let mae = (recon.clone() - x).abs().mean().into_scalar::<f32>();
    println!("mean abs error (in [-1,1] space): {mae}");
    println!("decode: {:?}", t.elapsed());

    // Side by side: original | reconstruction
    let out = to_image(recon);
    let mut combined = RgbaImage::new(w * 2, h);
    image::imageops::replace(&mut combined, &img, 0, 0);
    image::imageops::replace(&mut combined, &out, w as i64, 0);
    combined.save("roundtrip.png").unwrap();
}
