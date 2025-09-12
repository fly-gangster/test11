# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from lib.knn import KNearestNeighbor   # 确保这版不依赖 torch.utils.ffi

# =============== 四元数 → 旋转矩阵 ===============
def quat_to_rotmat(q, order='xyzw'):
    """
    q: (B, M, 4)
    order: 'xyzw'（默认）或 'wxyz'，按你的网络输出来设。
    返回: R (B, M, 3, 3)
    """
    # 先单位化，避免数值飘
    q = q / (q.norm(dim=2, keepdim=True) + 1e-8)

    if order == 'xyzw':
        qx, qy, qz, qw = q.unbind(dim=2)
    elif order == 'wxyz':
        qw, qx, qy, qz = q.unbind(dim=2)
    else:
        raise ValueError("Unsupported quaternion order")

    R11 = 1 - 2*(qy*qy + qz*qz)
    R12 = 2*(qx*qy - qz*qw)
    R13 = 2*(qx*qz + qy*qw)

    R21 = 2*(qx*qy + qz*qw)
    R22 = 1 - 2*(qx*qx + qz*qz)
    R23 = 2*(qy*qz - qx*qw)

    R31 = 2*(qx*qz - qy*qw)
    R32 = 2*(qy*qz + qx*qw)
    R33 = 1 - 2*(qx*qx + qy*qy)

    R = torch.stack([
        torch.stack([R11, R12, R13], dim=-1),
        torch.stack([R21, R22, R23], dim=-1),
        torch.stack([R31, R32, R33], dim=-1),
    ], dim=-2)  # (B, M, 3, 3)
    return R


# =============== 主损失（保留 DenseFusion 机制） ===============
def loss_calculation(pred_r, pred_t, pred_c, target, model_points, idx, points,
                     w, refine, num_point_mesh, sym_list, quat_order='xyzw'):
    """
    pred_r      : (B, M, 4)  四元数
    pred_t      : (B, M, 3)
    pred_c      : (B, M) or (B, M, 1)
    target      : (B, N, 3)  GT（相机坐标系）
    model_points: (B, N, 3)  模型点（物体坐标系）
    idx         : (B,)       物体 id
    points      : (B, M, 3)  输入场景点（相机系）
    w           : float
    refine      : bool
    num_point_mesh: N
    sym_list    : list[int]
    """
    device = pred_r.device
    B, M, _ = pred_r.shape
    N = num_point_mesh

    # 置信度形状与数值保护
    if pred_c.dim() == 3 and pred_c.size(-1) == 1:
        pred_c = pred_c.squeeze(-1)          # (B,M)
    pred_c = pred_c.clamp_min(1e-6)

    # 四元数 → 旋转矩阵
    R = quat_to_rotmat(pred_r, order=quat_order)           # (B,M,3,3)

    # 扩展到每点假设
    mp = model_points.unsqueeze(1).expand(B, M, N, 3)      # (B,M,N,3)
    tp = target.unsqueeze(1).expand(B, M, N, 3).contiguous()

    # 预测点云：R @ model + (points + t)
    mpR = torch.matmul(mp.unsqueeze(-2), R.unsqueeze(2)).squeeze(-2)   # (B,M,N,3)
    pred_xyz = mpR + (points.unsqueeze(2) + pred_t.unsqueeze(2))       # (B,M,N,3)

    # 对称物体（ADD-S）：仅非 refine 时做；逐 batch 处理，避免跨 batch 越界
    if not refine:
        knn = KNearestNeighbor(1)
        for b in range(B):
            if int(idx[b].item()) not in sym_list:
                continue
            # 展平成 (3, ·) 做 KNN
            t_flat = target[b].transpose(0, 1).contiguous()  # (3, N)
            p_flat = pred_xyz[b].reshape(-1, 3).transpose(0, 1).contiguous()  # (3, M*N)

            # 预测→GT 最近邻 —— 这一行要以 p_flat 为 query，才能返回 M*N 个索引
            inds = knn(p_flat.unsqueeze(0), t_flat.unsqueeze(0)).view(-1).to(
                device=device, dtype=torch.long
            )

            # 处理 1-based / 方向异常
            if inds.numel() == M * N and inds.min().item() == 1:
                inds = inds - 1
            if inds.numel() != M * N or inds.max().item() >= N or inds.min().item() < 0:
                # 极少数实现的参数语义相反，再尝试相反方向
                inds = knn(t_flat.unsqueeze(0), p_flat.unsqueeze(0)).view(-1).to(
                    device=device, dtype=torch.long
                )
                if inds.numel() == M * N and inds.min().item() == 1:
                    inds = inds - 1

            # 最终防越界
            inds = torch.clamp(inds, 0, N - 1)

            # 选列并还原 (M,N,3)
            t_sel = torch.index_select(t_flat, 1, inds).view(3, M, N).permute(1, 2, 0).contiguous()
            tp[b] = t_sel

    # 每点假设的平均点距 (B,M)
    dis_h = torch.norm(pred_xyz - tp, dim=3).mean(dim=2)   # (B,M)

    # 置信度损失
    loss = (dis_h * pred_c - w * torch.log(pred_c)).mean()

    # 选每个 batch 的最佳假设（按 c 最大），构造 refiner 坐标
    which_max = pred_c.argmax(dim=1)                                    # (B,)
    best_lin  = which_max + torch.arange(B, device=device) * M          # (B,)

    R_best = R.view(B * M, 3, 3)[best_lin]                              # (B,3,3)
    t_best = (points + pred_t).view(B * M, 3)[best_lin].unsqueeze(1)    # (B,1,3)

    # new_points/new_target（供 refiner）：把点、GT 平移到 t* 再右乘 R*
    pts_centered  = points - t_best                                     # (B,M,3)
    new_points = torch.matmul(pts_centered, R_best)  # (B,M,3) @ (B,3,3) → (B,M,3)
    tgt_centered  = target - t_best                                     # (B,N,3)
    new_target = torch.matmul(tgt_centered, R_best)  # (B,N,3) @ (B,3,3) → (B,N,3)

    dis_best = dis_h.gather(1, which_max.view(B, 1)).mean()

    return loss, dis_best, new_points.detach(), new_target.detach()


# 兼容原 train.py 的类封装
class Loss(nn.Module):
    def __init__(self, num_points_mesh, sym_list, quat_order='xyzw'):
        super().__init__()
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list
        self.quat_order = quat_order  # 'xyzw' 或 'wxyz'

    def forward(self, pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine):
        return loss_calculation(pred_r, pred_t, pred_c, target, model_points, idx, points,
                                w, refine, self.num_pt_mesh, self.sym_list, self.quat_order)
