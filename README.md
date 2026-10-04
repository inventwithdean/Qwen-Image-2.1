
# Qwen Image 2.1

This is a Rust implementation of the Qwen Image 2.1 which currently sits at Rank 1 of [Text-to-Image Arena](https://arena.ai/leaderboard/text-to-image?license=open-source) in Open Source category as of 04/10/26.

#### Sample Generation: 
![A sample image generated using this inference engine](./generations/main.png)




## Features
- Full BF16 weights for high fidelity generation
- RAM Layer Streaming for Low VRAM usage


## Components
- DiT (Completed)
- VAE (TODO)
- Text Encoder (TODO)

## License

Code distributed under MIT license. Weights are under [qwen-research license](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE) and not distributed with the repo.