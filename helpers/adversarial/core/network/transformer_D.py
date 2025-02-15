# adapted from https://github.com/AMD-AIG-AIMA/AMD-Diffusion-Distillation/blob/main/core/network/transformer_D.py
from typing import Callable
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel, FluxTransformerBlock, FluxSingleTransformerBlock
import torch.nn as nn
import torch
from torch.nn.utils.spectral_norm import SpectralNorm
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.fn(x) + x) / np.sqrt(2)


class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class BatchNormLocal(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 8, eps: float = 1e-5):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()

        # Reshape batch into groups.
        G = np.ceil(x.size(0)/self.virtual_bs).astype(int)
        x = x.view(G, -1, x.size(-2), x.size(-1))

        # Calculate stats.
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))

        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        return x.view(shape)


def make_block(channels: int, kernel_size: int) -> nn.Module:
    return nn.Sequential(
        SpectralConv1d(
            channels,
            channels,
            kernel_size = kernel_size,
            padding = kernel_size//2,
            padding_mode = 'circular',
        ),
        BatchNormLocal(channels),
        nn.LeakyReLU(0.2, True),
    )


# Adapted from https://github.com/autonomousvision/stylegan-t/blob/main/networks/discriminator.py
class DiscHead(nn.Module):
    def __init__(self, channels: int, c_dim: int, cmap_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.c_dim = c_dim
        self.cmap_dim = cmap_dim

        self.main = nn.Sequential(
            make_block(channels, kernel_size=1),
            ResidualBlock(make_block(channels, kernel_size=9))
        )

        if self.c_dim > 0:
            self.cmapper = nn.Linear(self.c_dim, cmap_dim)
            self.cls = SpectralConv1d(channels, cmap_dim, kernel_size=1, padding=0)
        else:
            self.cls = SpectralConv1d(channels, 1, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.main(x)
        out = self.cls(h)

        if self.c_dim > 0:
            cmap = self.cmapper(c).unsqueeze(-1)
            out = (out * cmap).sum(1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        return out

class FluxTransformer2DDiscriminator(nn.Module):
    """
    Discriminator that extracts intermediate features from a shared Flux transformer
    using PyTorch hooks. Features are extracted from attention layers of select transformer blocks (1, 15, 29, 43, 57).
    These features are processed through discriminator heads to produce adversarial scores.
    
    Key points:
    - Uses forward hooks to capture outputs from specific transformer blocks
    - Transformer is shared with generator, so we don't modify its components
    - Features are processed through DiscHead modules that use spectral normalization
    - Can extract from single layer or multiple layers (multiscale mode)
    """
    def __init__(self, transformer: FluxTransformer2DModel):
        super().__init__()
        self.transformer = transformer
        
        self.hooks = []
        # NOTE: Flux dev transformer has 57 blocks. 19 dual stream and 38 single stream.
        # since I want block indexes 0, 14, 28, 42, 56 and flux dev refers to the dual and single stream blocks separately
        # i want indexes 0, 14 from dual stream
        # single stream starts at absolute index 19
        # so I want single stream indexes 9, 23, 37

        self.dual_stream_indexes = [0, 14]
        self.single_stream_indexes = [9, 23, 37]

        # Storage for features
        self.features = []
        
        # hook appends module output into self.features
        def extract_features_hook(module, input, output):
            self.features.append(output)
            # apparently pytorch hooks can replace the forward output
            return output
            
        # attach hooks to the transformer blocks' attention layers at target indexes
        for idx in self.dual_stream_indexes:
            transformer_block = self.transformer.transformer_blocks[idx]
            assert isinstance(transformer_block, FluxTransformerBlock)
            hook = transformer_block.attn.register_forward_hook(extract_features_hook)
            self.hooks.append(hook)

        for idx in self.single_stream_indexes:
            transformer_block = self.transformer.single_transformer_blocks[idx]
            assert isinstance(transformer_block, FluxSingleTransformerBlock)
            hook = transformer_block.attn.register_forward_hook(extract_features_hook)
            self.hooks.append(hook)

        # discriminator heads
        num_hooks = len(self.dual_stream_indexes) + len(self.single_stream_indexes)        
        heads = []
        for i in range(num_hooks):
            heads.append(DiscHead(self.transformer.inner_dim, 0, 0))
        self.heads = nn.ModuleList(heads)

    @property
    def model(self):
        return self.transformer
    
    def remove_hooks(self):
        """Remove all hooks to prevent memory leaks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __del__(self):
        self.remove_hooks()
        
    """
    these two can be gotten from pipeline.prepare_latents:
    latents is initial noise
    latent_image_ids is Tensor of shape (num_patches, 3) that assigns XY coordinates to each image patch.

    timesteps should be float between 0 and 1

    these three can be gotten from pipelin encode_prompt:
    prompt_embeds is t5 prompt embedding
    pooled_prompt_embeds is CLIP pooled prompt embedding
    text_ids is a bunch of zeros, it seems.

    """
    def forward(self, latents, timesteps, prompt_embeds, 
                pooled_prompt_embeds, text_ids, latent_image_ids,
                guidance_scale=3.5,
                joint_attention_kwargs=None, added_cond_kwargs={}):
        # Clear features from previous forward passes
        self.features = []

        guidance = torch.full([1], guidance_scale) # in diffusers this was dtype fp32
        guidance = guidance.expand(latents.shape[0])
        
        # we have a hook that extracts the features
        self.transformer.forward(
            hidden_states=latents,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            timestep=timesteps, # may need to be divided by 1000. depending
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            guidance=guidance,
            joint_attention_kwargs=joint_attention_kwargs,
            *added_cond_kwargs
        )

        res_list = []
        for feat, head in zip(self.features, self.heads):
            res_list.append(head(feat.transpose(1,2), None).reshape(feat.shape[0], -1))
        
        concat_res = torch.cat(res_list, dim=1)
        
        return concat_res

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)