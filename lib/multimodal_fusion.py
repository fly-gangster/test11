# lib/multimodal_fusion.py
import os
import torch
import torch.nn as nn
from typing import Optional

# ---- 轻断言工具：确保张量为 (B,N,D) ----
DEBUG_SHAPE = os.getenv("DF_ASSERT", "1") == "1"

def _assert_bnD(x: torch.Tensor, name: str = "tensor", expected_D: Optional[int] = None):
    if not DEBUG_SHAPE:
        B, N, D = x.shape
        return B, N, D
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}")
    if x.dim() != 3:
        raise ValueError(f"{name} must be (B,N,D), got {tuple(x.shape)}")
    B, N, D = x.shape
    if B <= 0 or N <= 0 or D <= 0:
        raise ValueError(f"{name} has non-positive dims: {tuple(x.shape)}")
    if expected_D is not None and D != expected_D:
        raise ValueError(f"{name} D={D} != expected {expected_D}")
    return B, N, D

class _FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4, p: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(p),
        )

    def forward(self, x):  # (B,N,D)
        return self.net(x)

class _ChannelSpatialAttention(nn.Module):
    """通道-空间（token）注意力的轻量重标定。"""
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** -0.5
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):  # (B,N,D)
        attn = (x * self.scale).mean(-1, keepdim=True)  # (B,N,1)
        attn = torch.softmax(attn, dim=1)               # (B,N,1)
        out = x * attn                                  # (B,N,D)
        return self.gamma * out + x

class MultiModalFusion(nn.Module):
    r"""Dual-direction cross-attention fusion for RGB & point-cloud features.

    Inputs:
        f_rgb : (B, C_rgb, N)
        f_pc  : (B, C_pc,  N)
    Output:
        (B, embed_dim, N)
    """
    def __init__(
        self,
        rgb_channels: int = 128,
        pc_channels: int = 128,
        embed_dim: int = 256,
        heads: int = 4,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.proj_rgb = nn.Linear(rgb_channels, embed_dim, bias=False)
        self.proj_pc  = nn.Linear(pc_channels,  embed_dim, bias=False)
        self.attn_rgb_q = nn.MultiheadAttention(embed_dim, heads, dropout=drop)
        self.attn_pc_q  = nn.MultiheadAttention(embed_dim, heads, dropout=drop)
        self._bf = getattr(self.attn_rgb_q, "batch_first", False)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff    = _FeedForward(embed_dim, mlp_ratio=4, p=drop)
        self.ln_ff = nn.LayerNorm(embed_dim)
        self.csa   = _ChannelSpatialAttention(embed_dim)

    def forward(
        self,
        f_rgb: torch.Tensor,  # (B, C_rgb, N)
        f_pc:  torch.Tensor,  # (B, C_pc,  N)
        global_ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 输入检查
        if f_rgb.dim() != 3 or f_pc.dim() != 3:
            raise ValueError(f"Expect (B,C,N); got rgb={tuple(f_rgb.shape)}, pc={tuple(f_pc.shape)}")
        if f_rgb.size(2) != f_pc.size(2):
            raise ValueError(f"N mismatch: rgb.N={f_rgb.size(2)} vs pc.N={f_pc.size(2)}")

        # (B,N,Cin)
        rgb = f_rgb.permute(0, 2, 1)
        pc  = f_pc.permute(0, 2, 1)

        # 投影到 (B,N,D)
        rgb = self.proj_rgb(rgb)
        pc  = self.proj_pc(pc)

        # 形状断言
        D_rgb = getattr(self.attn_rgb_q, "embed_dim", None)
        D_pc  = getattr(self.attn_pc_q,  "embed_dim", None)
        Br, Nr, Dr = _assert_bnD(rgb, "rgb", expected_D=D_rgb)
        Bp, Np, Dp = _assert_bnD(pc,  "pc",  expected_D=D_pc)
        if Br != Bp or Nr != Np:
            raise ValueError(f"Batch/N mismatch after projection: rgb={Br,Nr,Dr}, pc={Bp,Np,Dp}")

        # RGB <- PC
        rgb_q = self.ln1(rgb)
        if self._bf:
            rgb_ca, _ = self.attn_rgb_q(rgb_q, pc, pc, need_weights=False)  # (B,N,D)
        else:
            rgb_ca_t, _ = self.attn_rgb_q(rgb_q.transpose(0, 1),  # (N,B,D)
                                          pc.transpose(0, 1),
                                          pc.transpose(0, 1),
                                          need_weights=False)
            rgb_ca = rgb_ca_t.transpose(0, 1)
        rgb_out = rgb + rgb_ca

        # PC <- RGB
        pc_q = self.ln1(pc)
        if self._bf:
            pc_ca, _ = self.attn_pc_q(pc_q, rgb, rgb, need_weights=False)  # (B,N,D)
        else:
            pc_ca_t, _ = self.attn_pc_q(pc_q.transpose(0, 1),
                                        rgb.transpose(0, 1),
                                        rgb.transpose(0, 1),
                                        need_weights=False)
            pc_ca = pc_ca_t.transpose(0, 1)
        pc_out = pc + pc_ca


        # 融合 + FFN + 重标定
        fuse = 0.5 * (rgb_out + pc_out)
        fuse = fuse + self.ff(self.ln2(fuse))
        fuse = self.csa(self.ln_ff(fuse))  # (B,N,D)

        return fuse.permute(0, 2, 1)       # (B,D,N)
