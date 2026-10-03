use burn::{
    Tensor,
    config::Config,
    module::{Module, Param},
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig1d},
    tensor::{
        Bool, Device, FloatDType, Int,
        activation::{gelu_approximate, silu},
        ops::PadMode,
        s,
    },
};

use crate::normalization::RMSNorm;

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

// in_channels = 256
// time_embed_dim = num_attention_head * attention_head_dim
// sample_proj_bias = False
// act_fn = silu
// out_dim = None
// post_act_fn = None
// cond_proj_dim = None
#[derive(Module, Debug)]
struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    fn forward(&self, mut sample: Tensor<2>) -> Tensor<2> {
        sample = self.linear_1.forward(sample);
        sample = silu(sample);
        self.linear_2.forward(sample)
    }
}

#[derive(Config, Debug)]
struct TimestepEmbeddingConfig {
    in_channels: usize,
    time_embed_dim: usize,
}

impl TimestepEmbeddingConfig {
    fn init(&self, device: &Device) -> TimestepEmbedding {
        let time_embed_dim_out = self.time_embed_dim;
        TimestepEmbedding {
            linear_1: LinearConfig::new(self.in_channels, self.time_embed_dim)
                .with_bias(false)
                .init(device),
            linear_2: LinearConfig::new(self.time_embed_dim, time_embed_dim_out)
                .with_bias(false)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
struct QwenImageTimestepProjEmbeddings {
    time_proj: QwenImageTemporalTimesteps,
    timestep_embedder: TimestepEmbedding,
}

impl QwenImageTimestepProjEmbeddings {
    fn forward(&self, timestep: Tensor<1>) -> Tensor<2> {
        let timesteps_proj = self.time_proj.forward(timestep);
        self.timestep_embedder.forward(timesteps_proj)
    }
}

#[derive(Config, Debug)]
struct QwenImageTimestepProjEmbeddingsConfig {
    embedding_dim: usize,
}

impl QwenImageTimestepProjEmbeddingsConfig {
    fn init(&self, device: &Device) -> QwenImageTimestepProjEmbeddings {
        QwenImageTimestepProjEmbeddings {
            time_proj: QwenImageTemporalTimestepsConfig::new(256).init(device),
            timestep_embedder: TimestepEmbeddingConfig::new(256, self.embedding_dim).init(device),
        }
    }
}

#[derive(Module, Debug)]
struct QwenImageZeroCenterRMSNorm {
    // (dim,)
    weight: Param<Tensor<1>>,
    eps: f64,
}

impl QwenImageZeroCenterRMSNorm {
    fn forward(&self, hidden_states: Tensor<3>) -> Tensor<3> {
        // hidden_states: (B, S, D)
        let arg = hidden_states.clone().powf_scalar(2.0).mean_dim(1) + self.eps;
        let rrms = arg.sqrt().recip();
        let weight = self
            .weight
            .val()
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(0)
            + 1.0; // (1, dim)
        hidden_states * rrms * weight
    }
}

#[derive(Config, Debug)]
struct QwenImageZeroCenterRMSNormConfig {
    dim: usize,
    eps: f64,
}

impl QwenImageZeroCenterRMSNormConfig {
    fn init(&self, device: &Device) -> QwenImageZeroCenterRMSNorm {
        QwenImageZeroCenterRMSNorm {
            weight: Param::from_tensor(Tensor::zeros([self.dim], device)),
            eps: self.eps,
        }
    }
}

// act = GELU(approximate=tanh)
#[derive(Module, Debug)]
struct QwenImageTextProjection {
    text_norm: QwenImageZeroCenterRMSNorm,
    in_layer: Linear,
    out_layer: Linear,
}

impl QwenImageTextProjection {
    fn forward(&self, hidden_states: Tensor<3>) -> Tensor<3> {
        let hidden_states = self.text_norm.forward(hidden_states);
        let hidden_states = self.in_layer.forward(hidden_states);
        let hidden_states = gelu_approximate(hidden_states);
        self.out_layer.forward(hidden_states)
    }
}

#[derive(Config, Debug)]
struct QwenImageTextProjectionConfig {
    context_in_dim: usize,
    hidden_size: usize,
    eps: f64,
}

impl QwenImageTextProjectionConfig {
    fn init(&self, device: &Device) -> QwenImageTextProjection {
        QwenImageTextProjection {
            text_norm: QwenImageZeroCenterRMSNormConfig::new(self.context_in_dim, self.eps)
                .init(device),
            in_layer: LinearConfig::new(self.context_in_dim, self.hidden_size)
                .with_bias(false)
                .init(device),
            out_layer: LinearConfig::new(self.hidden_size, self.hidden_size)
                .with_bias(false)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
struct QwenImageSwiGLUFeedForward {
    proj: Linear,
    out: Linear,
    gate_layer: Linear,
}

impl QwenImageSwiGLUFeedForward {
    fn forward(&self, hidden_states: Tensor<3>) -> Tensor<3> {
        let gate_out = silu(self.gate_layer.forward(hidden_states.clone()));
        let other = self.proj.forward(hidden_states);
        self.out.forward(gate_out * other)
    }
}

#[derive(Config, Debug)]
struct QwenImageSwiGLUFeedForwardConfig {
    hidden_size: usize,
    mlp_hidden_size: usize,
}

impl QwenImageSwiGLUFeedForwardConfig {
    fn init(&self, device: &Device) -> QwenImageSwiGLUFeedForward {
        QwenImageSwiGLUFeedForward {
            proj: LinearConfig::new(self.hidden_size, self.mlp_hidden_size)
                .with_bias(false)
                .init(device),
            out: LinearConfig::new(self.mlp_hidden_size, self.hidden_size)
                .with_bias(false)
                .init(device),
            gate_layer: LinearConfig::new(self.hidden_size, self.mlp_hidden_size)
                .with_bias(false)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
struct QwenImageAdaLayerNormContinuous {
    linear: Linear,
    // norm: LayerNorm, Burn doesn't supports elementwise_affine=False in LayerNorm as of now
    eps: f64,
}

impl QwenImageAdaLayerNormContinuous {
    fn forward(
        &self,
        hidden_states: Tensor<3>,
        conditioning_embedding: Tensor<2>,
        target_token_mask: Tensor<1, Bool>,
    ) -> Tensor<3> {
        // hidden_states: (B, S, D)
        // conditioning_embedding: (B+1, D)
        // target_token_mask: (S)

        // Manual parameter less LayerNorm
        let mean = hidden_states.clone().mean_dim(2); // (B, S, 1)
        let diff = hidden_states.clone() - mean; // (B, S, D)
        let var = diff.clone().powf_scalar(2.0).mean_dim(2); // (B, S, 1)
        let normalized_hidden_states = diff / (var + self.eps).sqrt(); // (B, S, D)

        let scale = self.linear.forward(silu(conditioning_embedding));
        let scale = select_modulation_rows(scale, target_token_mask);
        normalized_hidden_states * (1.0 + scale)
    }
}

fn select_modulation_rows(params: Tensor<2>, target_token_mask: Tensor<1, Bool>) -> Tensor<3> {
    // params: (B+1, D)
    // target_token_mask: (seq_len,)
    let [b, _d] = params.dims();
    let real = params.clone().slice(s![0..b - 1, ..]).unsqueeze_dim::<3>(1); // (B, 1, D)
    let zero = params.slice(s![b - 1.., ..]).unsqueeze_dim::<3>(0); // (1, 1, D)
    let target_token_mask = target_token_mask.unsqueeze_dims::<3>(&[0, 2]); // (1. seq_len, 1)
    zero.mask_where(target_token_mask, real)
}

#[derive(Config, Debug)]
struct QwenImageAdaLayerNormContinuousConfig {
    embedding_dim: usize,
    conditioning_embedding_dim: usize,
    eps: f64,
}

impl QwenImageAdaLayerNormContinuousConfig {
    fn init(&self, device: &Device) -> QwenImageAdaLayerNormContinuous {
        QwenImageAdaLayerNormContinuous {
            linear: LinearConfig::new(self.conditioning_embedding_dim, self.embedding_dim)
                .with_bias(false)
                .init(device),
            eps: self.eps,
        }
    }
}

fn qwenimage_prefix_segments(
    image_ids: Tensor<1, Int>,
    prefix_len: usize,
) -> Vec<(usize, usize, bool)> {
    // image_ids: (seq_len,)
    let prefix_ids = image_ids.slice(s![0..prefix_len]); // (prefix_len,)
    let prefix_ids = prefix_ids.into_data().try_into_vec::<i64>().unwrap();
    let mut segments = vec![];
    let mut start = 0;
    for index in 1..=prefix_len {
        if index == prefix_len || prefix_ids[index] != prefix_ids[start] {
            segments.push((start, index, prefix_ids[start] < 0));
            start = index;
        }
    }
    segments
}

/// Stores the K and V projecctions (post-RoPE) for the prefix extracted at the first denoising step.
/// K and V both are of shape: (batch_size, num_prefix_tokens, num_heads, head_dim)
#[derive(Clone)]
struct QwenImageKVLayerCache {
    k: Tensor<4>,
    v: Tensor<4>,
}

impl QwenImageKVLayerCache {
    fn store(&mut self, k: Tensor<4>, v: Tensor<4>) {
        self.k = k;
        self.v = v;
    }

    fn get(&self) -> (Tensor<4>, Tensor<4>) {
        (self.k.clone(), self.v.clone())
    }
}

struct QwenImageKVCache {
    layer_caches: Vec<QwenImageKVLayerCache>,
}

impl QwenImageKVCache {
    fn get_layer(&self, layer_idx: usize) -> QwenImageKVLayerCache {
        self.layer_caches[layer_idx].clone()
    }
}

enum KvCacheMode {
    EXTRACT,
    CACHED,
}

#[derive(Module, Debug)]
struct QwenImageAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Vec<Linear>, // (Original ModuleList was [Linear, Dropout] but dropout isn't used in inference and we need the name mapping to work, so using Vec<Linear>)
    norm_q: RMSNorm,
    norm_k: RMSNorm,
}

fn attention(
    query: Tensor<4>,
    key: Tensor<4>,
    value: Tensor<4>,
    attention_mask: Option<Tensor<4, Bool>>,
) -> Tensor<4> {
    Tensor::zeros([3, 3, 3, 3], &query.device())
}

impl QwenImageAttention {
    fn forward(
        &self,
        hidden_states: Tensor<3>,
        attention_mask: Option<Tensor<4, Bool>>,
        rotary_emb: (Tensor<2>, Tensor<2>),
        layer_cache: Option<&mut QwenImageKVLayerCache>,
        kv_cache_mode: Option<KvCacheMode>,
        prefix_len: usize,
        segments: Option<Vec<(usize, usize, bool)>>,
        key_valid: Option<Tensor<2, Bool>>,
    ) -> Tensor<3> {
        let num_attention_heads = 32;
        let head_dim = 128;
        let [b, s, _] = hidden_states.dims();
        let query = self.to_q.forward(hidden_states.clone()); // (B, S, 4096)
        let key = self.to_k.forward(hidden_states.clone()); // (B, S, 4096)
        let value = self.to_v.forward(hidden_states.clone()); // (B, S, 4096)

        let mut query = query.reshape([b, s, num_attention_heads, head_dim]);
        let mut key = key.reshape([b, s, num_attention_heads, head_dim]);
        let mut value = value.reshape([b, s, num_attention_heads, head_dim]);
        // All are of shape (B, S, 32, 128) now

        query = self.norm_q.forward(query);
        key = self.norm_k.forward(key);

        query = apply_rotary_emb_qwen(query, rotary_emb.clone());
        key = apply_rotary_emb_qwen(key, rotary_emb);
        if let (Some(layer_cache), Some(kv_cache_mode)) = (layer_cache, kv_cache_mode) {
            match kv_cache_mode {
                KvCacheMode::EXTRACT => {
                    layer_cache.store(
                        key.clone().slice(s![.., ..prefix_len, .., ..]),
                        value.clone().slice(s![.., ..prefix_len, .., ..]),
                    );
                }
                KvCacheMode::CACHED => {
                    let (cached_k, cached_v) = layer_cache.get();
                    key = Tensor::cat(vec![cached_k, key], 1);
                    value = Tensor::cat(vec![cached_v, value], 1);
                }
            }
        }
        let seq_len_q = query.dims()[1];
        let seq_len_kv = key.dims()[1];

        let hidden_states = if let Some(segments) = segments {
            let prefix_len = segments.last().unwrap().1;
            let mut outputs = vec![];
            for (start, end, is_text) in segments {
                let mut seg_mask = None;
                if is_text {
                    let seg_len = end - start;
                    // (1, 1, seg_len, end)
                    seg_mask = Some(
                        Tensor::cat(
                            vec![
                                Tensor::<2, Int>::ones([seg_len, start], &query.device()).bool(),
                                Tensor::<2, Int>::ones([seg_len, seg_len], &query.device())
                                    .tril(0)
                                    .bool(),
                            ],
                            1,
                        )
                        .unsqueeze_dims::<4>(&[0, 1]),
                    );
                }
                if let Some(key_valid) = key_valid.clone() {
                    let mut seg_key_valid = key_valid.unsqueeze_dims::<4>(&[1, 2]); // (B, 1, 1, S)
                    seg_key_valid = seg_key_valid.slice(s![.., .., .., 0..end]);
                    seg_mask = match seg_mask {
                        Some(mask) => Some(mask.bool_and(seg_key_valid)),
                        None => Some(seg_key_valid),
                    }
                }

                outputs.push(attention(
                    query.clone().slice(s![.., start..end, .., ..]),
                    key.clone().slice(s![.., 0..end, .., ..]),
                    value.clone().slice(s![.., 0..end, .., ..]),
                    seg_mask,
                ));
            }
            // (B, 1, 1, S)
            let trailing_mask = key_valid.map(|kv| kv.unsqueeze_dims::<4>(&[1, 2]));
            outputs.push(attention(
                query.slice(s![.., prefix_len.., .., ..]),
                key,
                value,
                trailing_mask,
            ));

            let prefill_hidden_states = Tensor::cat(outputs, 1);
            prefill_hidden_states.slice(s![.., 0..seq_len_q, .., ..]) // (B, S, 32, 128)
        } else {
            let decode_hidden_states = attention(query, key, value, attention_mask);
            decode_hidden_states.slice(s![.., 0..seq_len_q, .., ..])
        };

        let hidden_states = hidden_states.reshape([b, seq_len_q, num_attention_heads * head_dim]);
        let hidden_states = self.to_out[0].forward(hidden_states);

        hidden_states
    }
}
