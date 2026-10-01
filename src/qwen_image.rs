use burn::{
    Tensor,
    config::Config,
    module::Module,
    tensor::{Device, FloatDType},
};

// use_real_unbind_dim = -1
fn apply_rotary_emb_qwen(x: Tensor<4>, freqs_cis: (Tensor<2>, Tensor<2>)) -> Tensor<4> {
    // x: (Batch, Sequence, Heads, Dimension)
    // cos: (S, D)
    // sin: (S, D)

    let [b, s, h, d] = x.dims();
    let (cos, sin) = freqs_cis;
    let cos = cos.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(2); // (1, S, D) then (1, S, 1, D)
    let sin = sin.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(2); // (1, S, D) then (1, S, 1, D)

    let x_reshaped = x.clone().reshape([b, s, h, d / 2, 2]); // (Assuming d is exactly divisible by 2)
    let x_real = x_reshaped.clone().slice_dim(4, 0).squeeze_dim::<4>(4); // (B, S, H, D/2)
    let x_imag = x_reshaped.clone().slice_dim(4, 1).squeeze_dim::<4>(4); // (B, S, H, D/2)
    let x_rotated = Tensor::stack::<5>(vec![-x_imag, x_real], 4); // (B, S, H, D/2, 2)
    let x_rotated = x_rotated.reshape([b, s, h, d]);

    x * cos + x_rotated * sin // (B, S, H, D)
}

// timestep_dim = 256
// max_period = 10_000
// time_factor = 1_000.0
#[derive(Module, Debug)]
struct QwenImageTemporalTimesteps {
    // (half,) or (128,)
    freqs: Tensor<1>,
    time_factor: f64,
}

impl QwenImageTemporalTimesteps {
    fn forward(&self, timestep: Tensor<1>) -> Tensor<2> {
        // timestep: (B,)
        let timestep = self.time_factor * timestep;
        let timestep = timestep.unsqueeze_dim::<2>(1); // (B, 1)
        let freqs = self.freqs.clone().unsqueeze_dim::<2>(0); // (1, half)
        let args = timestep * freqs;
        let cos = args.clone().cos(); // (B, half)
        let sin = args.clone().sin(); // (B, half)
        Tensor::cat(vec![cos, sin], 1) // (B, timestep_dim) as half * 2 = timestep_dim
    }
}

#[derive(Config, Debug)]
struct QwenImageTemporalTimestepsConfig {
    #[config(default = 256)]
    timestep_dim: usize,
    #[config(default = 10_000)]
    max_period: usize,
    #[config(default = 1_000.0)]
    time_factor: f64,
}

impl QwenImageTemporalTimestepsConfig {
    fn init(&self, device: &Device) -> QwenImageTemporalTimesteps {
        let half = (self.timestep_dim / 2) as f64;
        let freqs = -(self.max_period as f64).ln()
            * Tensor::arange(0..half as i64, device).cast(FloatDType::F32)
            / half;
        let freqs = freqs.exp();
        QwenImageTemporalTimesteps {
            freqs,
            time_factor: self.time_factor,
        }
    }
}


