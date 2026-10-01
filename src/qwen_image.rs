use burn::Tensor;

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

    x * cos + x_rotated * sin
}
