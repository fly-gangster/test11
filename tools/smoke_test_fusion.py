# tools/smoke_test_fusion.py
import _init_paths
import os, sys, torch

# 强制不缓冲
sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("DF_ASSERT", "1")

from lib.multimodal_fusion import MultiModalFusion
from lib.cross_attention import DualCrossEncoder  # 你的文件名

def run_multimodal_fusion(device="cpu"):
    print(f"\n[MultiModalFusion] device={device}", flush=True)
    B, N, C_rgb, C_pc, D = 2, 128, 128, 128, 256
    x_rgb = torch.randn(B, C_rgb, N, device=device)
    x_pc  = torch.randn(B, C_pc , N, device=device)
    m = MultiModalFusion(rgb_channels=C_rgb, pc_channels=C_pc, embed_dim=D, heads=4).to(device)
    m.eval()
    with torch.no_grad():
        y = m(x_rgb, x_pc)
    print("OK shape:", tuple(y.shape), flush=True)

def run_dual_cross_encoder(device="cpu"):
    print(f"\n[DualCrossEncoder] device={device}", flush=True)
    B, Np, Ni, C = 2, 256, 96, 64
    pc  = torch.randn(B, Np, C, device=device)
    img = torch.randn(B, Ni, C, device=device)
    enc = DualCrossEncoder(dim=C, heads=4, depth=2).to(device)
    enc.eval()
    with torch.no_grad():
        pc2, img2 = enc(pc, img)
    print("OK shapes:", tuple(pc2.shape), tuple(img2.shape), flush=True)

def main():
    print(">>> Smoke test started", flush=True)
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for d in devs:
        run_multimodal_fusion(d)
        run_dual_cross_encoder(d)
    print("\nAll smoke tests finished.", flush=True)

if __name__ == "__main__":
    main()
