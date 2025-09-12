# tools/train_refine.py
import os
import time
import math
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

# --- 项目内模块 ---
import _init_paths  # 确保 import lib/*
from lib.network import PoseNet, PoseRefineNet
from lib.loss_refiner import Loss_refine
from datasets.linemod.dataset import PoseDataset as PoseDataset_LM
# 若训练 YCB：from datasets.ycb.dataset import PoseDataset as PoseDataset_YCB

def _to_tensor(x):
    import numpy as np
    if torch.is_tensor(x):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x

def _ensure_bdim(x):
    """若没有 batch 维，则在最前面补 1。"""
    if not torch.is_tensor(x):
        return x
    if x.dim() == 0:
        return x.view(1, 1)  # 标量兜底
    if x.dim() == 1:
        return x.unsqueeze(0)  # (N) -> (1,N)
    if x.dim() == 2 and x.size(-1) == 3:
        return x.unsqueeze(0)  # (N,3) -> (1,N,3)
    if x.dim() == 3 and x.size(0) == 3:
        return x.unsqueeze(0)  # (3,H,W) -> (1,3,H,W)
    return x  # 已经有 batch 维的情况

def _unpack_batch(data, num_points, device=None):
    """
    支持 dict / tuple / list，且兼容单样本无 batch 维的返回：
      points: (N,3) 或 (B,N,3)
      choose: (1,N) / (N) / (B,1,N) / (B,N)
      img   : (3,H,W) 或 (B,3,H,W)
      target/model_points: (M,3) 或 (B,M,3)
      idx   : (1,) / (B,) / (B,1)
    输出统一为：
      points(B,N,3), choose(B,1,N), img(B,3,H,W),
      target(B,M,3), model_points(B,M,3), idx(B,)
    """
    # 1) 统一收集
    items = []
    if isinstance(data, dict):
        for k, v in data.items():
            items.append((k.lower(), _to_tensor(v)))
    elif isinstance(data, (list, tuple)):
        for i, v in enumerate(data):
            items.append((f"_{i}", _to_tensor(v)))
    else:
        raise RuntimeError(f"[unpack] unsupported batch type: {type(data)}")

    # 2) 先按常见位次抓取（你的观察顺序： _0 points, _1 choose, _2 img, _3 target, _4 model_points, _5 idx）
    keymap = {k: v for k, v in items}

    def _get(name_or_idx, fallbacks=()):
        if name_or_idx in keymap and torch.is_tensor(keymap[name_or_idx]):
            return keymap[name_or_idx]
        for fb in fallbacks:
            if fb in keymap and torch.is_tensor(keymap[fb]):
                return keymap[fb]
        return None

    img    = _get("_2", ("img", "image", "rgb"))
    pts    = _get("_0", ("points", "pts", "cloud", "point_cloud"))
    choose = _get("_1", ("choose", "indices", "index"))
    tgt    = _get("_3", ("target", "gt_points", "gt"))
    mp     = _get("_4", ("model_points", "model", "modelpts"))
    idx    = _get("_5", ("idx", "label", "obj", "object_id"))

    # 3) 若以上没拿到，再按形状兜底搜索
    def _first(cond):
        for k, v in items:
            if torch.is_tensor(v):
                try:
                    if cond(v): return v
                except Exception:
                    pass
        return None

    if img is None:
        img = _first(lambda t: (t.dim()==4 and t.size(1)==3) or (t.dim()==3 and t.size(0)==3))
    if pts is None:
        pts = _first(lambda t: (t.dim()==3 and t.size(-1)==3) or (t.dim()==2 and t.size(-1)==3))
    if choose is None:
        choose = _first(lambda t: t.dim() in (1,2,3) and t.dtype in (torch.int32, torch.int64))
    if idx is None:
        idx = _first(lambda t: t.dim() in (1,2) and t.dtype in (torch.int32, torch.int64))
    if tgt is None or mp is None:
        cand_3d = [v for _, v in items if torch.is_tensor(v) and (
            (v.dim()==3 and v.size(-1)==3) or (v.dim()==2 and v.size(-1)==3)
        )]
        # 去掉已识别为 pts 的那个
        cand = [t for t in cand_3d if t is not pts]
        if len(cand) >= 2:
            # 先都补 batch，再按 M 排序
            candB = [t if t.dim()==3 else t.unsqueeze(0) for t in cand]  # (B,M,3)
            candB = sorted(candB, key=lambda t: t.size(1))
            tgt = tgt or candB[-1]
            mp  = mp  or candB[0]
        elif len(cand) == 1:
            tgt = tgt or (cand[0] if cand[0].dim()==3 else cand[0].unsqueeze(0))

    # 4) 统一补 batch 维
    img = _ensure_bdim(img)
    pts = _ensure_bdim(pts)
    tgt = _ensure_bdim(tgt) if tgt is not None else None
    mp  = _ensure_bdim(mp)  if mp  is not None else None

    # choose：可能是 (N) / (1,N) / (B,N) / (B,1,N)
    if choose is None:
        raise RuntimeError(f"[unpack] Missing choose. Observed: {[ (k, (tuple(v.shape) if torch.is_tensor(v) else type(v))) for k,v in items ]}")
    if choose.dim() == 1:
        choose = choose.view(1, 1, -1)
    elif choose.dim() == 2:
        # (1,N) 或 (B,N) -> (B,1,N)
        if pts.dim() == 3 and choose.size(0) != pts.size(0):
            # 绝大多数是 (1,N)，补成 (1,1,N)
            choose = choose.unsqueeze(0)
        else:
            choose = choose.unsqueeze(1)
    elif choose.dim() == 3 and choose.size(1) != 1:
        # (B,N,1) -> (B,1,N)
        if choose.size(2) == 1:
            choose = choose.transpose(1, 2)
        else:
            # 非预期形状，尽量变成 (B,1,N)
            choose = choose[:, :1, :]

    # idx： (1,) / (B,) / (B,1) 统一成 (B,)
    if idx is None:
        raise RuntimeError(f"[unpack] Missing idx. Observed: {[ (k, (tuple(v.shape) if torch.is_tensor(v) else type(v))) for k,v in items ]}")
    if idx.dim() == 2 and idx.size(1) == 1:
        idx = idx.view(-1)
    elif idx.dim() == 1:
        idx = idx.view(-1)
    else:
        # 其它形状尽量 squeeze 成 (B,)
        idx = idx.view(-1)

    # 兜底：若 target/model_points 任一缺失，用另一个替代
    if tgt is None and mp is not None: tgt = mp
    if mp  is None and tgt is not None: mp  = tgt

    # 5) 最终缺失检查
    missing = []
    for name, t in (("img", img), ("points", pts), ("choose", choose), ("idx", idx), ("target", tgt), ("model_points", mp)):
        if t is None: missing.append(name)
    if missing:
        shapes = [(k, (tuple(v.shape) if torch.is_tensor(v) else type(v))) for k, v in items]
        raise RuntimeError(f"[unpack] Missing {missing}. Observed: {shapes}")

    # 6) 迁移设备
    if device is not None:
        img = img.to(device, non_blocking=True)
        pts = pts.to(device, non_blocking=True)
        choose = choose.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        mp  = mp.to(device, non_blocking=True)

    return pts, choose, img, tgt, mp, idx

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_loader(dataset_root: str, num_points: int, workers: int, batch_size: int):
    from torch.utils.data import DataLoader
    # LineMOD
    train_ds = PoseDataset_LM('train', num_points, False, dataset_root, 0.0, False)
    val_ds   = PoseDataset_LM('eval',  num_points, False, dataset_root, 0.0, True)

    def _collate_single(batch):
        # batch 是长度=1 的 list，直接返回里面那份原始样本
        return batch[0]

    collate = _collate_single if batch_size == 1 else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=True,
        collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=False,
        collate_fn=collate
    )
    return train_ds, val_ds, train_loader, val_loader



@torch.no_grad()
def pick_pose_from_tokens(pr, pt, pc, points):
    """
    与一阶段一致：从逐点预测里按最大置信度挑选 (q0, t0)。
    pr: (B,N,4) unit quats; pt: (B,N,3); pc: (B,N,1); points: (B,N,3)
    返回：q0:(B,4), t0:(B,3)
    """
    B, N, _ = pr.shape
    conf = pc.view(B, N)             # (B,N)
    sel = conf.argmax(dim=1)         # (B,)
    batch = torch.arange(B, device=pr.device)
    q0 = pr[batch, sel]              # (B,4)
    # DenseFusion 的 t = 选中点坐标 + offset
    t_offsets = pt[batch, sel]       # (B,3)
    pts_sel = points[batch, sel]     # (B,3)
    t0 = pts_sel + t_offsets
    return q0, t0


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
    w, x, y, z = q.unbind(dim=1)
    B = q.size(0)
    R = q.new_empty(B, 3, 3)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def validate(posenet, refiner, loader, criterion, device, num_points):
    posenet.eval(); refiner.eval()
    loss_meter, dis_meter, n_samples = 0.0, 0.0, 0
    printed = False

    for data in loader:
        points, choose, img, target, model_points, idx = _unpack_batch(
            data, num_points=num_points, device=device
        )
        idx = idx.view(-1)

        # Debug 一次性打印形状
        if not printed:
            print("[debug] shapes:",
                  "img", tuple(img.shape),
                  "points", tuple(points.shape),
                  "choose", tuple(choose.shape),
                  "target", tuple(target.shape),
                  "model_points", tuple(model_points.shape),
                  "idx", tuple(idx.shape))
            printed = True

        with torch.no_grad():
            pr, pt, pc, emb = posenet(img=img, points=points, choose=choose, obj=idx.view(-1, 1))
            pr = pr / (pr.norm(dim=2, keepdim=True) + 1e-8)
            q0, t0 = pick_pose_from_tokens(pr, pt, pc, points)
            R0 = quat_to_mat(q0)
            new_points = torch.bmm(points - t0.unsqueeze(1), R0)

        dq, dt = refiner(new_points, emb, idx)
        dq = dq / (dq.norm(dim=1, keepdim=True) + 1e-8)

        # 内部合成姿态后度量（按上一姿态对齐）
        loss, dists = criterion(dq, dt, model_points, target, idx, R_prev=R0, t_prev=t0)
        bs = points.size(0)
        loss_meter += float(loss.item()) * bs
        dis_meter  += float(dists.mean().item()) * bs
        n_samples  += bs

    return loss_meter / max(1, n_samples), dis_meter / max(1, n_samples)


def train_one_epoch(posenet, refiner, loader, criterion, optimizer, device, num_points, clip_grad=1.0):
    posenet.eval(); refiner.train()
    loss_meter = 0.0
    printed = False

    for data in loader:
        points, choose, img, target, model_points, idx = _unpack_batch(
            data, num_points=num_points, device=device
        )
        idx = idx.view(-1)

        if not printed:
            print("[debug] shapes:",
                  "img", tuple(img.shape),
                  "points", tuple(points.shape),
                  "choose", tuple(choose.shape),
                  "target", tuple(target.shape),
                  "model_points", tuple(model_points.shape),
                  "idx", tuple(idx.shape))
            printed = True

        with torch.no_grad():
            pr, pt, pc, emb = posenet(img=img, points=points, choose=choose, obj=idx.view(-1, 1))
            pr = pr / (pr.norm(dim=2, keepdim=True) + 1e-8)
            q0, t0 = pick_pose_from_tokens(pr, pt, pc, points)
            R0 = quat_to_mat(q0)
            new_points = torch.bmm(points - t0.unsqueeze(1), R0)  # (B,N,3)

        dq, dt = refiner(new_points, emb, idx)              # (B,4),(B,3)
        dq = dq / (dq.norm(dim=1, keepdim=True) + 1e-8)

        loss, _ = criterion(dq, dt, model_points, target, idx, R_prev=R0, t_prev=t0)

        optimizer.zero_grad()          # 兼容老版 PyTorch
        loss.backward()
        if clip_grad is not None and clip_grad > 0:
            nn.utils.clip_grad_norm_(refiner.parameters(), clip_grad)
        optimizer.step()

        loss_meter += float(loss.item()) * points.size(0)

    return loss_meter / len(loader.dataset)


def main():
    ap = argparse.ArgumentParser("Train RefineNet (Stage-2)")
    ap.add_argument("--dataset_root", type=str, required=True,
                    help="Path to Linemod_preprocessed directory")
    ap.add_argument("--posenet_ckpt", type=str, required=True,
                    help="Path to pre-trained PoseNet weights (*.pth)")
    ap.add_argument("--save_dir", type=str, default="experiments/refine_linemod")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=1)   # 建议先用 1 跑通
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--num_points", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # --- Data ---
    train_ds, val_ds, train_loader, val_loader = build_loader(
        args.dataset_root, args.num_points, args.workers, args.batch_size
    )
    sym_list = train_ds.get_sym_list()
    num_points_mesh = train_ds.get_num_points_mesh()

    # --- Models ---
    posenet = PoseNet(num_points=args.num_points, num_obj=13).to(device)  # LineMOD 13 类

    # ===== 兼容加载 PoseNet 权重（宽松/过滤分类头/处理前缀） =====
    ckpt = torch.load(args.posenet_ckpt, map_location=device)
    state = ckpt.get("state_dict", ckpt)

    mapped = {}
    for k, v in state.items():
        k2 = k
        # 去掉最前面的 "module."
        if k2.startswith("module."):
            k2 = k2[len("module."):]
        # ckpt 里可能是 "cnn.model.module."，统一成 "cnn.model."
        k2 = k2.replace("cnn.model.module.", "cnn.model.")
        mapped[k2] = v

    # 过滤分类头（如果有）
    filtered = {k: v for k, v in mapped.items()
                if not (k.startswith("cnn.model.final.") or k.startswith("final."))}

    missing_unexp = posenet.load_state_dict(filtered, strict=False)
    print("[PoseNet] load_state_dict strict=False")
    print("  #missing keys   :", len(missing_unexp.missing_keys))
    print("  #unexpected keys:", len(missing_unexp.unexpected_keys))
    # ===============================================================

    posenet.eval()
    for p in posenet.parameters():
        p.requires_grad = False

    refiner = PoseRefineNet(num_points=args.num_points, num_obj=13).to(device)

    # --- Loss & Optim ---
    criterion = Loss_refine(num_points_mesh, sym_list, reduction="mean").to(device)
    optimizer = torch.optim.Adam(refiner.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val_dis = math.inf
    ckpt_best = os.path.join(args.save_dir, "refiner_best.pth")
    ckpt_last = os.path.join(args.save_dir, "refiner_last.pth")

    print(f"Start training RefineNet for {args.epochs} epochs")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(posenet, refiner, train_loader, criterion, optimizer, device, args.num_points)

        val_loss, val_dis = validate(posenet, refiner, val_loader, criterion, device, args.num_points)

        dt = time.time() - t0
        print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_dis {val_dis:.4f} | {dt:.1f}s")

        # 保存最后一次
        torch.save(refiner.state_dict(), ckpt_last)
        # 按 val_dis 选最佳
        if val_dis < best_val_dis:
            best_val_dis = val_dis
            torch.save(refiner.state_dict(), ckpt_best)
            print(f"  ↳ New best val_dis={best_val_dis:.4f}  (saved to {ckpt_best})")

        # 简单 lr 衰减（每 20 epochs ×0.1）
        if epoch in (20, 40):
            for g in optimizer.param_groups:
                g["lr"] *= 0.1
            print(f"  ↳ LR decay: now lr={optimizer.param_groups[0]['lr']:.1e}")

    print("Done.")


if __name__ == "__main__":
    main()
