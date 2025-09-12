# lib/network.py (cleaned & aligned with latest modules)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from lib.pspnet import PSPNet
from lib.cross_attention import DualCrossEncoder
from lib.multimodal_fusion import MultiModalFusion


def _gather_rgb_features(feat: torch.Tensor, choose: torch.Tensor) -> torch.Tensor:
    """
    feat  : (B, C, H, W)  — PSPNet feature map after upsampling
    choose: (B, 1, N)     — flattened spatial indices (0..H*W-1)
    return: (B, C, N)
    """
    B, C, H, W = feat.shape
    N = choose.size(-1)
    feat_flat = feat.view(B, C, H * W)
    # clamp to be safe
    choose = choose.long().clamp_(0, H * W - 1)
    idx = choose.expand(B, C, N)  # (B, C, N)
    rgb = torch.gather(feat_flat, 2, idx)  # (B, C, N)
    return rgb


class PoseNet(nn.Module):
    """
    Lightweight DenseFusion-style PoseNet with:
      - PSPNet RGB backbone (output feat ~ 64ch at input resolution)
      - Conv1d RGB reduce 64->128
      - Conv1d PC project 3->128
      - DualCrossEncoder(dim=128, heads=2, depth=1, drop=0.1) (token-level, bidirectional)
      - MultiModalFusion(rgb=128, pc=128, embed_dim=256, heads=4, drop=0.1)
      - Per-point heads for r(4), t(3), c(1)
    """
    def __init__(self, num_points: int, num_obj: int):
        super().__init__()
        self.num_points = int(num_points)
        self.num_obj = int(num_obj)

        # 1) RGB encoder
        self.cnn = PSPNet(n_classes=50)  # n_classes is irrelevant here; we only want features

        # 2) Channel projections
        self.rgb_reduce = nn.Conv1d(64, 128, 1)  # (B,64,N) -> (B,128,N)
        self.pc_proj   = nn.Conv1d(3,  128, 1)   # (B,3,N)  -> (B,128,N)

        # 3) Token-level cross-attention (lightweight)
        self.cross_enc = DualCrossEncoder(dim=128, heads=2, depth=1, drop=0.1)

        # 4) Feature fusion (returns (B, D, N) with D=embed_dim)
        self.fuse = MultiModalFusion(
            rgb_channels=128,
            pc_channels=128,
            embed_dim=256,
            heads=4,
            drop=0.1,
        )

        # 5) Heads
        in_ch = 256 + 128  # fused(256) + global(128)
        self.conv1_r = nn.Conv1d(in_ch, 640, 1)
        self.conv1_t = nn.Conv1d(in_ch, 640, 1)
        self.conv1_c = nn.Conv1d(in_ch, 640, 1)

        self.conv2_r = nn.Conv1d(640, 256, 1)
        self.conv2_t = nn.Conv1d(640, 256, 1)
        self.conv2_c = nn.Conv1d(640, 256, 1)

        self.conv3_r = nn.Conv1d(256, 4, 1)
        self.conv3_t = nn.Conv1d(256, 3, 1)
        self.conv3_c = nn.Conv1d(256, 1, 1)

        # init
        for m in [self.rgb_reduce, self.pc_proj,
                  self.conv1_r, self.conv1_t, self.conv1_c,
                  self.conv2_r, self.conv2_t, self.conv2_c,
                  self.conv3_r, self.conv3_t, self.conv3_c]:
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    @torch.no_grad()
    def _encode_rgb(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (B,3,H,W)
        return: feat (B,64,H,W)  — PSPNet final feature before logits (upsampled)
        """
        logits, feat = self.cnn(img, return_features=True)
        return feat  # (B,64,H,W)

    def forward(self, img: torch.Tensor, points: torch.Tensor, choose: torch.Tensor, idx: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        img   : (B,3,H,W)
        points: (B,M,3)
        choose: (B,1,M) with M==num_points
        idx   : (B,)
        returns: pred_r(B,M,4), pred_t(B,M,3), pred_c(B,M,1), emb(B,M,256)
        """
        B, M, _ = points.shape

        # ---- RGB branch
        feat = self._encode_rgb(img)                      # (B,64,H,W) at input res
        rgb = _gather_rgb_features(feat, choose)          # (B,64,M)
        rgb = self.rgb_reduce(rgb)                        # (B,128,M)  ← 只降维一次

        # ---- Point branch
        pc = points.transpose(1, 2).contiguous()          # (B,3,M)
        pc = self.pc_proj(pc)                             # (B,128,M)

        # ---- Cross-encoding (token-level, (B,N,C))
        rgb_tok = rgb.permute(0, 2, 1).contiguous()       # (B,M,128)
        pc_tok  = pc.permute(0, 2, 1).contiguous()        # (B,M,128)
        pc_tok, rgb_tok = self.cross_enc(pc_tok, rgb_tok) # (B,M,128), (B,M,128)

        # back to (B,C,N)
        rgb = rgb_tok.permute(0, 2, 1).contiguous()       # (B,128,M)
        pc  = pc_tok.permute(0, 2, 1).contiguous()        # (B,128,M)

        # ---- Fusion → (B,256,M)
        fuse = self.fuse(rgb, pc)                         # (B,256,M)

        # ---- Global context (PC branch)
        g = torch.max(pc, dim=2, keepdim=True)[0]         # (B,128,1)
        g = g.expand(-1, -1, M)                           # (B,128,M)

        feat_all = torch.cat([fuse, g], dim=1)            # (B,384,M)

        # ---- Heads
        rx = F.relu(self.conv1_r(feat_all))
        tx = F.relu(self.conv1_t(feat_all))
        cx = F.relu(self.conv1_c(feat_all))

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))
        cx = F.relu(self.conv2_c(cx))

        pred_r = self.conv3_r(rx).permute(0, 2, 1).contiguous()  # (B,M,4)
        pred_t = self.conv3_t(tx).permute(0, 2, 1).contiguous()  # (B,M,3)
        pred_c = torch.sigmoid(self.conv3_c(cx)).permute(0, 2, 1).contiguous()  # (B,M,1)

        # emb for refiner (use fused tokens in (B,M,256))
        emb = fuse.permute(0, 2, 1).contiguous()          # (B,M,256)
        return pred_r, pred_t, pred_c, emb


class PoseRefineNet(nn.Module):
    """
    Simple refiner that takes new_points(B,M,3) and per-point embedding (B,M,256),
    and outputs delta rotations/translations per point.
    """
    def __init__(self, num_points: int, num_obj: int):
        super().__init__()
        self.num_points = int(num_points)
        self.num_obj = int(num_obj)

        self.pc_proj = nn.Conv1d(3, 128, 1)   # (B,3,M)->(B,128,M)
        in_ch = 128 + 256                      # concat with emb(B,256,M)
        self.conv1_r = nn.Conv1d(in_ch, 640, 1)
        self.conv1_t = nn.Conv1d(in_ch, 640, 1)
        self.conv2_r = nn.Conv1d(640, 256, 1)
        self.conv2_t = nn.Conv1d(640, 256, 1)
        self.conv3_r = nn.Conv1d(256, 4, 1)
        self.conv3_t = nn.Conv1d(256, 3, 1)

        for m in [self.pc_proj, self.conv1_r, self.conv1_t, self.conv2_r, self.conv2_t, self.conv3_r, self.conv3_t]:
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, new_points: torch.Tensor, emb: torch.Tensor, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        new_points: (B,M,3)
        emb       : (B,M,256)
        idx       : (B,)  (unused here; kept for API compatibility)
        returns   : pred_r(B,M,4), pred_t(B,M,3)
        """
        pc = new_points.transpose(1, 2).contiguous()      # (B,3,M)
        pc = self.pc_proj(pc)                             # (B,128,M)

        embT = emb.transpose(1, 2).contiguous()           # (B,256,M)
        feat = torch.cat([pc, embT], dim=1)               # (B,384,M)

        rx = F.relu(self.conv1_r(feat))
        tx = F.relu(self.conv1_t(feat))
        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))
        pred_r = self.conv3_r(rx).permute(0, 2, 1).contiguous()  # (B,M,4)
        pred_t = self.conv3_t(tx).permute(0, 2, 1).contiguous()  # (B,M,3)
        return pred_r, pred_t
