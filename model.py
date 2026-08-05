"""
Masked Autoencoder (MAE) implementation based on:
"Masked Autoencoders Are Scalable Vision Learners"
He et al., ICCV 2021
"""

import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import Block, PatchEmbed, get_2d_sincos_pos_embed
import numpy as np


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with ViT encoder and lightweight decoder."""
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        mask_ratio=0.75,
    ):
        """
        Args:
            img_size: Input image size
            patch_size: Patch size
            in_chans: Number of input channels
            embed_dim: Encoder embedding dimension
            depth: Encoder depth (number of transformer blocks)
            num_heads: Encoder number of attention heads
            decoder_embed_dim: Decoder embedding dimension (default 512 for <10% compute per token)
            decoder_depth: Decoder depth (default 8 transformer blocks)
            decoder_num_heads: Decoder number of attention heads
            mlp_ratio: MLP hidden dim ratio
            norm_layer: Normalization layer
            norm_pix_loss: Whether to normalize pixel loss
            mask_ratio: Proportion of patches to mask (default 0.75)
        """
        super().__init__()
        
        # MAE parameters
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Class token and positional embeddings for encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )
        
        # Encoder blocks
        self.blocks = nn.ModuleList([
            Block(
                embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                norm_layer=norm_layer
            )
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        
        # Decoder embeddings
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            Block(
                decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,
                norm_layer=norm_layer
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        # Reconstruction head: project to pixel values
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size ** 2 * in_chans, bias=True
        )
        
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        # Positional embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5),
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5),
            cls_token=True
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )
        
        # Initialize other parameters
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights_helper)
        
    def _init_weights_helper(self, m):
        """Helper for weight initialization."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        
        Args:
            x: (N, L, D) tensor of shape (batch, num_patches, embed_dim)
            mask_ratio: Masking ratio (default 0.75)
        
        Returns:
            x_masked: Masked input
            mask: Binary mask (1 is masked, 0 is kept)
            ids_restore: Indices to restore original order
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        # Generate the binary mask: 0 is keep, 1 is mask
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x):
        """
        Encoder forward pass.
        
        Args:
            x: (N, C, H, W) input images
        
        Returns:
            x: Encoded representation
            mask: Binary mask of masked patches
            ids_restore: Indices to restore original order
        """
        # Patch embedding
        x = self.patch_embed(x)  # (N, L, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed[:, 1:, :]
        
        # Mask patches
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        
        # Prepend class token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply encoder blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        """
        Decoder forward pass.
        
        Args:
            x: Encoded representation from encoder
            ids_restore: Indices to restore original order
        
        Returns:
            x: Reconstructed patches
        """
        # Project to decoder dimension
        x = self.decoder_embed(x)
        
        # Append mask tokens for masked patches
        mask_tokens = self.mask_token.expand(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], -1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # No class token
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2])
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)  # Append class token
        
        # Add positional embedding
        x = x + self.decoder_pos_embed
        
        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # Predict pixels
        x = self.decoder_pred(x)
        
        # Remove class token
        x = x[:, 1:, :]
        
        return x
    
    def forward_loss(self, imgs, pred, mask):
        """
        Compute MAE loss.
        
        Args:
            imgs: Original images (N, C, H, W)
            pred: Predicted patches (N, L, patch_size^2 * C)
            mask: Binary mask (N, L)
        
        Returns:
            loss: Scalar loss
        """
        # Patchify target
        target = self.patchify(imgs)  # (N, L, patch_size^2 * C)
        
        if self.norm_pix_loss:
            # Normalize target per patch
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        
        # NOTE: MSE loss used as per paper specification
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (N, L)
        
        # Only compute loss on masked patches
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def patchify(self, imgs):
        """
        Convert images to patch embeddings.
        
        Args:
            imgs: (N, C, H, W) images
        
        Returns:
            x: (N, L, patch_size^2 * C) patches
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        
        return x
    
    def unpatchify(self, x):
        """
        Convert patches back to images.
        
        Args:
            x: (N, L, patch_size^2 * C) patches
        
        Returns:
            imgs: (N, C, H, W) images
        """
        p = self.patch_size
        h = w = int((x.shape[1]) ** 0.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        
        return imgs
    
    def forward(self, imgs):
        """
        Forward pass: encode, mask, decode, compute loss.
        
        Args:
            imgs: (N, C, H, W) input images
        
        Returns:
            loss: Scalar loss
            pred: Predicted patches for visualization
            mask: Binary mask for visualization
        """
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        
        return loss, pred, mask


def mae_vit_base(**kwargs):
    """MAE with ViT-Base encoder (12 blocks, 768 dim)."""
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def mae_vit_large(**kwargs):
    """MAE with ViT-Large encoder (24 blocks, 1024 dim)."""
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def mae_vit_huge(**kwargs):
    """MAE with ViT-Huge encoder (32 blocks, 1280 dim)."""
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model