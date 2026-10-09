use burn::{
    Tensor,
    config::Config,
    module::Module,
    tensor::{Device, activation::silu},
};

use crate::vae::vae_modules::{
    QwenImageCausalConv3D, QwenImageCausalConv3DConfig, QwenImageMidBlock, QwenImageMidBlockConfig,
    QwenImageRMSNorm3D, QwenImageRMSNorm3DConfig, QwenImageResidualDownBlock,
    QwenImageResidualDownBlockConfig, QwenImageResidualUpBlock, QwenImageResidualUpBlockConfig,
};

// is_residual = true
#[derive(Module, Debug)]
pub struct QwenImageEncoder3D {
    conv_in: QwenImageCausalConv3D,
    down_blocks: Vec<QwenImageResidualDownBlock>,
    mid_block: QwenImageMidBlock,
    norm_out: QwenImageRMSNorm3D,
    conv_out: QwenImageCausalConv3D,
}

impl QwenImageEncoder3D {
    pub fn forward(&self, x: Tensor<5>) -> Tensor<5> {
        let mut x = self.conv_in.forward(x);
        for block in &self.down_blocks {
            x = block.forward(x);
        }

        x = self.mid_block.forward(x);
        x = self.norm_out.forward(x);
        x = silu(x);
        self.conv_out.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct QwenImageEncoder3DConfig {
    in_channels: usize,
    dim: usize,
    z_dim: usize,
    dim_mult: Vec<usize>,
    num_res_blocks: usize,
    temporal_downsample: Vec<bool>,
}

impl QwenImageEncoder3DConfig {
    pub fn init(&self, device: &Device) -> QwenImageEncoder3D {
        let mut dims = vec![self.dim];
        for mult in &self.dim_mult {
            dims.push(self.dim * mult);
        }
        let conv_in = QwenImageCausalConv3DConfig::new(self.in_channels, dims[0], [3, 3])
            .with_padding([1, 1])
            .init(device);

        let mut down_blocks = vec![];
        for i in 0..self.dim_mult.len() {
            let in_dim = dims[i];
            let out_dim = dims[i + 1];
            let down_flag = i != self.dim_mult.len() - 1;
            let temporal_downsample = if down_flag {
                self.temporal_downsample[i]
            } else {
                false
            };
            down_blocks.push(
                QwenImageResidualDownBlockConfig::new(in_dim, out_dim, self.num_res_blocks)
                    .with_down_flag(down_flag)
                    .with_temporal_downsample(temporal_downsample)
                    .init(device),
            );
        }

        let out_dim = dims.last().expect("dims shouldn't be empty");
        let mid_block = QwenImageMidBlockConfig::new(*out_dim)
            .with_num_layers(1)
            .init(device);

        let norm_out = QwenImageRMSNorm3DConfig::new(*out_dim).init(device);
        let conv_out = QwenImageCausalConv3DConfig::new(*out_dim, self.z_dim, [3, 3])
            .with_padding([1, 1])
            .init(device);

        QwenImageEncoder3D {
            conv_in,
            down_blocks,
            mid_block,
            norm_out,
            conv_out,
        }
    }
}

// is_residual = true
#[derive(Module, Debug)]
pub struct QwenImageDecoder3D {
    conv_in: QwenImageCausalConv3D,
    mid_block: QwenImageMidBlock,
    up_blocks: Vec<QwenImageResidualUpBlock>,
    norm_out: QwenImageRMSNorm3D,
    conv_out: QwenImageCausalConv3D,
}

impl QwenImageDecoder3D {
    pub fn forward(&self, x: Tensor<5>) -> Tensor<5> {
        let mut x = self.conv_in.forward(x);
        x = self.mid_block.forward(x);
        for block in &self.up_blocks {
            x = block.forward(x);
        }
        x = self.norm_out.forward(x);
        x = silu(x);
        self.conv_out.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct QwenImageDecoder3DConfig {
    dim: usize,
    z_dim: usize,
    dim_mult: Vec<usize>,
    num_res_blocks: usize,
    temporal_upsample: Vec<bool>,
    out_channels: usize,
}

impl QwenImageDecoder3DConfig {
    pub fn init(&self, device: &Device) -> QwenImageDecoder3D {
        let mut dims = vec![self.dim * self.dim_mult.last().expect("dim_mult should not be empty")];
        for mult in self.dim_mult.iter().rev() {
            dims.push(self.dim * mult);
        }
        let conv_in = QwenImageCausalConv3DConfig::new(self.z_dim, dims[0], [3, 3])
            .with_padding([1, 1])
            .init(device);
        let mid_block = QwenImageMidBlockConfig::new(dims[0])
            .with_num_layers(1)
            .init(device);
        let mut up_blocks = vec![];
        for i in 0..self.dim_mult.len() {
            let in_dim = dims[i];
            let out_dim = dims[i + 1];
            let up_flag = i != self.dim_mult.len() - 1;
            let temporal_upsample = if up_flag {
                self.temporal_upsample[i]
            } else {
                false
            };
            up_blocks.push(
                QwenImageResidualUpBlockConfig::new(in_dim, out_dim, self.num_res_blocks)
                    .with_up_flag(up_flag)
                    .with_temporal_upsample(temporal_upsample)
                    .init(device),
            );
        }
        let out_dim = dims.last().expect("dims should not be empty!");
        let norm_out = QwenImageRMSNorm3DConfig::new(*out_dim).init(device);
        let conv_out = QwenImageCausalConv3DConfig::new(*out_dim, self.out_channels, [3, 3])
            .with_padding([1, 1])
            .init(device);

        QwenImageDecoder3D {
            conv_in,
            mid_block,
            up_blocks,
            norm_out,
            conv_out,
        }
    }
}
