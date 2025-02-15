I'm working on implementing a large-scale (12B parameter) GAN with a diffusion model backbone. I'd like to implement the R1 gradient penalty approximation from the Seaweed-APT paper. Here are the key details:

Original R1 regularization: 
- LR1 = ||∇xD(x, c)||²₂
- Requires higher-order gradients (double backward) which doesn't work with FSDP

Seaweed's approximation:
- LaR1 = ||D(x, c) - D(N(x, σI), c)||²₂ 
- Where N(x, σI) means adding small Gaussian noise to x
- Uses σ = 0.01 for images, σ = 0.1 for videos
- Applied with weight λ = 100
- Applied every discriminator training step

I need help implementing this in PyTorch in a way that:
1. Works with FSDP and gradient checkpointing
2. Can be easily added to the discriminator loss
3. Handles both conditional and unconditional cases
4. Maintains numerical stability

Could you help write a PyTorch implementation with proper testing assertions to verify correctness?