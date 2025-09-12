# lib/cross_attention.py
import os
import torch
import torch.nn as nn
from typing import Optional

# ---- 轻断言：确保 (B,N,C) ----
DEBUG_SHAPE = os.getenv("DF_ASSERT", "1") == "1"
def _assert_bnC(x: torch.Tensor, name: str = "tensor", expected_C: Optional[int] = None):
    if not DEBUG_SHAPE:
        B, N, C = x.shape
        return B, N, C
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}")
    if x.dim() != 3:
        raise ValueError(f"{name} must be (B,N,C), got {tuple(x.shape)}")
    B, N, C = x.shape
    if B <= 0 or N <= 0 or C <= 0:
        raise ValueError(f"{name} has non-positive dims: {tuple(x.shape)}")
    if expected_C is not None and C != expected_C:
        raise ValueError(f"{name} C={C} != expected {expected_C}")
    return B, N, C


class CrossAttnBlock(nn.Module):
    def __init__(self, dim, heads: int = 4, drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.ln_q  = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)

        # 不用 batch_first；统一按老接口 (L,N,E) 调用
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop)

        self.ln_out = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(dim * 4, dim), nn.Dropout(drop),
        )

    def forward(self, q, kv):  # 输入 (B,N,C)
        Bq, Nq, Cq = _assert_bnC(q,  "q",  expected_C=self.dim)
        Bk, Nk, Ck = _assert_bnC(kv, "kv", expected_C=self.dim)
        if Bq != Bk:
            raise ValueError(f"Batch mismatch: q.B={Bq}, kv.B={Bk}")
        if Cq != Ck:
            raise ValueError(f"Channel mismatch: q.C={Cq}, kv.C={Ck}")

        q2  = self.ln_q(q)    # (B,N,C)
        kv2 = self.ln_kv(kv)  # (B,N,C)

        # 统一走转置路径： (B,N,C) -> (N,B,C) -> attn -> (B,N,C)
        q2t, kv2t = q2.transpose(0, 1), kv2.transpose(0, 1)       # (N,B,C)
        attn_t, _ = self.attn(q2t, kv2t, kv2t, need_weights=False) # (N,B,C)
        attn_out  = attn_t.transpose(0, 1)                         # (B,N,C)

        x = q + attn_out
        x = x + self.mlp(self.ln_out(x))
        return x


# 例：DualCrossEncoder 的构造
class DualCrossEncoder(nn.Module):
    def __init__(self, dim=128, heads=2, depth=1, attn_drop=0.1, proj_drop=0.1, ffn_ratio=2, drop=None):
        if drop is not None:
            attn_drop = proj_drop = drop
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):  # depth=1
            self.layers.append(nn.ModuleDict(dict(
                ln_q = nn.LayerNorm(dim),
                ln_kv= nn.LayerNorm(dim),
                attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=attn_drop),  # heads=2
                ffn  = nn.Sequential(
                    nn.Linear(dim, dim*ffn_ratio), nn.ReLU(inplace=True),
                    nn.Dropout(proj_drop),
                    nn.Linear(dim*ffn_ratio, dim), nn.Dropout(proj_drop),
                )
            )))

    def forward(self, pc_feat, img_feat):  # 约定输入是 (B,N,C)
        # 如果你的 MHA 期望 (N,B,C)，这里要 transpose；保持和你原代码一致即可
        q = pc_feat    # (B,N,C)
        k = img_feat   # (B,N,C)
        for blk in self.layers:
            # 下面这几行保持你原来维度处理一致（必要时 transpose）
            qn  = blk['ln_q'](q)
            kvn = blk['ln_kv'](k)
            # 如果你的 MultiheadAttention 不是 batch_first，就做 (N,B,C) 转置
            qT, kT, vT = qn.transpose(0,1), kvn.transpose(0,1), kvn.transpose(0,1)
            out, _ = blk['attn'](qT, kT, vT)
            out = out.transpose(0,1)
            q = q + out                         # 残差
            q = q + blk['ffn'](q)               # FFN 残差
            # 交换方向（img 作为 Q，pc 作为 K/V）
            q, k = k, q
        # 最后再交换回来，保持 (pc_out, img_out) 顺序
        return k, q

