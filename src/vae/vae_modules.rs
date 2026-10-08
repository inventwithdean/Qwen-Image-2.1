use burn::{
    Tensor,
    config::Config,
    module::{Module, Param},
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{
        Device,
        activation::silu,
        module::{attention, interpolate},
        ops::{AttentionModuleOptions, InterpolateMode, InterpolateOptions, PadMode},
        s,
    },
};
use serde::{Deserialize, Serialize};

#[derive(Module, Debug)]
pub struct QwenImageAvgDown3D {
    in_channels: usize,
    out_channels: usize,
    factor_t: usize,
    factor_s: usize,
    factor: usize,
    group_size: usize,
}

impl QwenImageAvgDown3D {
    pub fn forward(&self, x: Tensor<5>) -> Tensor<5> {
        // x: (B, C, T, H, W)
        let pad_t = (self.factor_t - x.dims()[2] % self.factor_t) % self.factor_t;
        let x = x.pad([(pad_t, 0), (0, 0), (0, 0)], PadMode::Constant(0.0));
        let [b, c, t, h, w] = x.dims();
        let x = x.reshape([
            b,
            c,
            t / self.factor_t,
            self.factor_t,
            h / self.factor_s,
            self.factor_s,
            w / self.factor_s,
            self.factor_s,
        ]);
        let x = x.permute([0, 1, 3, 5, 7, 2, 4, 6]);
        let x = x.reshape([
            b,
            c * self.factor,
            t / self.factor_t,
            h / self.factor_s,
            w / self.factor_s,
        ]);
        let x = x.reshape([
            b,
            self.out_channels,
            self.group_size,
            t / self.factor_t,
            h / self.factor_s,
            w / self.factor_s,
        ]);

        x.mean_dim(2).squeeze_dim(2)
    }
}

#[derive(Config, Debug)]
pub struct QwenImageAvgDown3DConfig {
    in_channels: usize,
    out_channels: usize,
    factor_t: usize,
    #[config(default = 1)]
    factor_s: usize,
}

impl QwenImageAvgDown3DConfig {
    pub fn init(&self) -> QwenImageAvgDown3D {
        let factor = self.factor_t * self.factor_s * self.factor_s;
        assert!(
            (self.in_channels * factor % self.out_channels == 0),
            "`in_channels` ({}) times the downsampling factor ({}) must be divisible by `out_channels` ({}).",
            self.in_channels,
            factor,
            self.out_channels
        );

        let group_size = self.in_channels * factor / self.out_channels;
        QwenImageAvgDown3D {
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            factor_t: self.factor_t,
            factor_s: self.factor_s,
            factor,
            group_size,
        }
    }
}

#[derive(Module, Debug)]
pub struct QwenImageDupUp3D {
    in_channels: usize,
    out_channels: usize,
    factor_t: usize,
    factor_s: usize,
    factor: usize,
    repeats: usize,
}

impl QwenImageDupUp3D {
    pub fn forward(&self, x: Tensor<5>, first_chunk: bool) -> Tensor<5> {
        let [b, c, t, h, w] = x.dims();

        let x = x
            .unsqueeze_dim::<6>(2) // (B, C, 1, T, H, W)
            .expand([b, c, self.repeats, t, h, w])
            .reshape([b, c * self.repeats, t, h, w]); // (B, C * repeats, t, h, w)

        let x = x.reshape([
            b,
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            t,
            h,
            w,
        ]);

        let x = x.permute([0, 1, 5, 2, 6, 3, 7, 4]); // (B, out_channels, t, factor_t, h, factor_s, w, factor_s)
        let new_t = t * self.factor_t;
        let new_h = h * self.factor_s;
        let new_w = w * self.factor_s;
        let x = x.reshape([b, self.out_channels, new_t, new_h, new_w]);

        if first_chunk {
            x.slice(s![.., .., self.factor_t - 1.., .., ..])
        } else {
            x
        }
    }
}

#[derive(Config, Debug)]
pub struct QwenImageDupUp3DConfig {
    in_channels: usize,
    out_channels: usize,
    factor_t: usize,
    #[config(default = 1)]
    factor_s: usize,
}

impl QwenImageDupUp3DConfig {
    pub fn init(&self) -> QwenImageDupUp3D {
        let factor = self.factor_t * self.factor_s * self.factor_s;
        assert!(self.out_channels * factor % self.in_channels == 0);
        let repeats = self.out_channels * factor / self.in_channels;
        QwenImageDupUp3D {
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            factor_t: self.factor_t,
            factor_s: self.factor_s,
            factor,
            repeats,
        }
    }
}

#[derive(Module, Debug)]
pub struct QwenImageCausalConv3D {
    pub conv2d: Conv2d,
    padding: [usize; 2],
}

impl QwenImageCausalConv3D {
    pub fn forward(&self, x: Tensor<5>) -> Tensor<5> {
        let x = x.squeeze_dim::<4>(2); // (B, C, H, W)
        let [pad_h, pad_w] = self.padding;
        let x = x.pad([(pad_h, pad_h), (pad_w, pad_w)], PadMode::Constant(0.0));
        let x = self.conv2d.forward(x);
        x.unsqueeze_dim(2)
    }
}

#[derive(Config, Debug)]
pub struct QwenImageCausalConv3DConfig {
    in_channels: usize,
    out_channels: usize,
    kernel_size: [usize; 2],
    #[config(default = "[1, 1]")]
    stride: [usize; 2],
    #[config(default = "[0, 0]")]
    padding: [usize; 2],
}

impl QwenImageCausalConv3DConfig {
    pub fn init(&self, device: &Device) -> QwenImageCausalConv3D {
        let conv2d = Conv2dConfig::new([self.in_channels, self.out_channels], self.kernel_size)
            .with_stride(self.stride)
            .init(device);
        QwenImageCausalConv3D {
            conv2d,
            padding: self.padding,
        }
    }
}

#[derive(Module, Debug)]
pub struct QwenImageRMSNorm {
    gamma: Param<Tensor<1>>,
    bias: Option<Param<Tensor<1>>>,
    scale: f32,
    dim: usize,
}

impl QwenImageRMSNorm {
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let norm = x
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1)
            .sqrt()
            .clamp_min(1e-12);
        let mut out = (x / norm) * self.scale;

        let mut shape = [1; D]; // If D=5, [1, dim, 1, 1, 1]
        shape[1] = self.dim;

        let gamma = self.gamma.val().reshape(shape);
        out = out * gamma;
        if let Some(bias) = &self.bias {
            let bias = bias.val().reshape(shape);
            out = out + bias;
        }
        out
    }
}

#[derive(Config, Debug)]
pub struct QwenImageRMSNormConfig {
    pub dim: usize,
    #[config(default = false)]
    pub bias: bool,
}

impl QwenImageRMSNormConfig {
    pub fn init(&self, device: &Device) -> QwenImageRMSNorm {
        let scale = (self.dim as f32).sqrt();
        let gamma = Param::from_tensor(Tensor::ones([self.dim], device));
        let bias = if self.bias {
            Some(Param::from_tensor(Tensor::zeros([self.dim], device)))
        } else {
            None
        };

        QwenImageRMSNorm {
            gamma,
            bias,
            scale,
            dim: self.dim,
        }
    }
}

// scale_factor is always [2.0, 2.0] with nearest-exact mode
#[derive(Module, Debug)]
pub struct QwenImageUpsample {}

impl QwenImageUpsample {
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        interpolate(
            x,
            InterpolateOptions::new(InterpolateMode::Nearest).with_scale_factor([2.0, 2.0]),
        )
    }
}

#[derive(Config, Debug)]
pub struct QwenImageUpsampleConfig {}

impl QwenImageUpsampleConfig {
    pub fn init(&self) -> QwenImageUpsample {
        QwenImageUpsample {}
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum ResampleMode {
    Upsample2D,
    Upsample3D,
    Downsample2D,
    Downsample3D,
}

// We don't need time_conv
#[derive(Module, Debug)]
struct QwenImageResample {
    resample: (Option<QwenImageUpsample>, Conv2d),
}

impl QwenImageResample {
    pub fn forward(&self, x: Tensor<5>) -> Tensor<5> {
        let [b, c, t, h, w] = x.dims();
        let x = x.permute([0, 2, 1, 3, 4]); // (B, T, C, H, W)
        let mut x = x.reshape([b * t, c, h, w]); // (B * T, C, H, W)
        if let Some(upsample) = &self.resample.0 {
            // Upsample
            x = upsample.forward(x);
        } else {
            // Downsample
            x = x.pad([(0, 1), (0, 1)], PadMode::Constant(0.0));
        }
        x = self.resample.1.forward(x);
        let [_, c, h, w] = x.dims();
        let x = x.reshape([b, t, c, h, w]); // (B, T, C, H, W)
        let x = x.permute([0, 2, 1, 3, 4]); // (B, C, T, H, W)
        x
    }
}

#[derive(Config, Debug)]
pub struct QwenImageResampleConfig {
    dim: usize,
    mode: ResampleMode,
    upsample_out_dim: Option<usize>,
}

impl QwenImageResampleConfig {
    pub fn init(&self, device: &Device) -> QwenImageResample {
        let upsample_out_dim = self.upsample_out_dim.unwrap_or(self.dim / 2);
        let resample = match self.mode {
            ResampleMode::Upsample2D => (
                Some(QwenImageUpsampleConfig::new().init()),
                Conv2dConfig::new([self.dim, upsample_out_dim], [3, 3])
                    .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                    .init(device),
            ),
            ResampleMode::Upsample3D => (
                Some(QwenImageUpsampleConfig::new().init()),
                Conv2dConfig::new([self.dim, upsample_out_dim], [3, 3])
                    .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                    .init(device),
            ),
            ResampleMode::Downsample2D => (
                None,
                Conv2dConfig::new([self.dim, self.dim], [3, 3])
                    .with_stride([2, 2])
                    .init(device),
            ),
            ResampleMode::Downsample3D => (
                None,
                Conv2dConfig::new([self.dim, self.dim], [3, 3])
                    .with_stride([2, 2])
                    .init(device),
            ),
        };
        QwenImageResample { resample }
    }
}

#[derive(Module, Debug)]
pub struct QwenImageResidualBlock {
    norm1: QwenImageRMSNorm,
    conv1: QwenImageCausalConv3D,
    norm2: QwenImageRMSNorm,
    conv2: QwenImageCausalConv3D,
    conv_shortcut: Option<QwenImageCausalConv3D>,
}

impl QwenImageResidualBlock {
    pub fn forward(&self, x: Tensor<5>) -> Tensor<5> {
        // x: (B, C, T, H, W)
        let h = match &self.conv_shortcut {
            Some(shortcut) => shortcut.forward(x.clone()),
            None => x.clone(),
        };

        let x = self.norm1.forward(x);
        let x = silu(x);
        let x = self.conv1.forward(x);

        let x = self.norm2.forward(x);
        let x = silu(x);
        let x = self.conv2.forward(x);
        x + h
    }
}

#[derive(Config, Debug)]
pub struct QwenImageResidualBlockConfig {
    in_dim: usize,
    out_dim: usize,
}

impl QwenImageResidualBlockConfig {
    pub fn init(&self, device: &Device) -> QwenImageResidualBlock {
        let conv_shortcut = if self.in_dim != self.out_dim {
            Some(QwenImageCausalConv3DConfig::new(self.in_dim, self.out_dim, [1, 1]).init(device))
        } else {
            None
        };
        QwenImageResidualBlock {
            norm1: QwenImageRMSNormConfig::new(self.in_dim).init(device),
            conv1: QwenImageCausalConv3DConfig::new(self.in_dim, self.out_dim, [3, 3])
                .with_padding([1, 1])
                .init(device),
            norm2: QwenImageRMSNormConfig::new(self.out_dim).init(device),
            conv2: QwenImageCausalConv3DConfig::new(self.out_dim, self.out_dim, [3, 3])
                .with_padding([1, 1])
                .init(device),
            conv_shortcut,
        }
    }
}

#[derive(Module, Debug)]
pub struct QwenImageAttentionBlock {
    norm: QwenImageRMSNorm,
    to_qkv: Conv2d,
    proj: Conv2d,
}

impl QwenImageAttentionBlock {
    pub fn forward(&self, x: Tensor<5>) -> Tensor<5> {
        // x: (B, C, T, H, W)
        let identity = x.clone();
        let [b, c, t, h, w] = x.dims();

        let x = x.permute([0, 2, 1, 3, 4]); // (B, T, C, H, W)
        let x = x.reshape([b * t, c, h, w]); // (B * T, C, H, W)
        let x = self.norm.forward(x);

        let qkv = self.to_qkv.forward(x); // (B * T, C, H, W)
        let qkv = qkv.reshape([b * t, 1, c * 3, h * w]);
        let qkv = qkv.permute([0, 1, 3, 2]); // (B * T, 1, H * W, C * 3)
        let qkv_chunks = qkv.chunk(3, 3);
        let (q, k, v) = (
            qkv_chunks[0].clone(),
            qkv_chunks[1].clone(),
            qkv_chunks[2].clone(),
        );
        // Each of q, k and v is now (B * T, 1, H * W, C)
        // expects (batch_size, num_heads, seq_len, head_dim)
        let x = attention(q, k, v, None, None, AttentionModuleOptions::default());
        let x = x.squeeze_dim::<3>(1); // (B * T, H * W, C)
        let x = x.permute([0, 2, 1]); // (B * T, C, H * W)
        let x = x.reshape([b * t, c, h, w]); // (B * T, C, H, W)

        let x = self.proj.forward(x);
        let x = x.reshape([b, t, c, h, w]);
        let x = x.permute([0, 2, 1, 3, 4]); // (B, C, T, H, W)
        x + identity
    }
}

#[derive(Config, Debug)]
pub struct QwenImageAttentionBlockConfig {
    dim: usize,
}

impl QwenImageAttentionBlockConfig {
    pub fn init(&self, device: &Device) -> QwenImageAttentionBlock {
        QwenImageAttentionBlock {
            norm: QwenImageRMSNormConfig::new(self.dim).init(device),
            to_qkv: Conv2dConfig::new([self.dim, self.dim * 3], [1, 1]).init(device),
            proj: Conv2dConfig::new([self.dim, self.dim], [1, 1]).init(device),
        }
    }
}
