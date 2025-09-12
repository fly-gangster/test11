# lib/loss_refiner.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 工具：四元数 → 旋转矩阵 ----------
def _quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """q: (B,4) (w,x,y,z) → R: (B,3,3). 自动归一化。"""
    q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
    w, x, y, z = q.unbind(dim=1)
    B = q.size(0)
    R = q.new_empty(B, 3, 3)
    # 参考标准公式
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

class Loss_refine(nn.Module):
    """
    细化阶段的 ADD/ADD-S 损失。
    支持两种调用方式：
      1) 传“增量姿态” + 上一帧姿态（推荐，align by previous）：
         forward(pred_r, pred_t, model_points, target, idx,
                 R_prev=..., t_prev=...)
         其中 pred_r/pred_t 是 ΔR/Δt；内部会合成 R = R_prev @ ΔR, t = t_prev + Δt
      2) 直接传“绝对姿态”（兼容旧代码）：
         forward(pred_r, pred_t, model_points, target, idx)
         此时 pred_r/pred_t 被视为最终姿态。
    其余：
      - model_points: (B, M, 3)  物体模型点（单位：米）
      - target      : (B, M, 3)  GT 点（R_gt X + t_gt）
      - idx         : (B,)       物体 id
      - sym_list    : list[int]  对称物体 id（与数据集一致）
    """
    def __init__(self, num_points_mesh, sym_list, reduction: str = "mean"):
        super().__init__()
        self.num_points_mesh = num_points_mesh
        self.sym_list = set(int(x) for x in sym_list)
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction

    @torch.no_grad()
    def _is_symmetric(self, idx_b: int) -> bool:
        return (idx_b in self.sym_list)

    def _compose_with_prev(self,
                           dR_q: torch.Tensor,  # (B,4)  Δq
                           dt:   torch.Tensor,  # (B,3)  Δt
                           R_prev: torch.Tensor,  # (B,3,3)
                           t_prev: torch.Tensor   # (B,3)
                           ):
        """合成 R = R_prev @ ΔR,  t = t_prev + Δt"""
        dR = _quat_to_mat(dR_q)                         # (B,3,3)
        R  = torch.bmm(R_prev, dR)                      # (B,3,3)
        t  = t_prev + dt                                # (B,3)
        return R, t

    def forward(self,
                pred_r: torch.Tensor,         # (B,4) 或 Δq
                pred_t: torch.Tensor,         # (B,3) 或 Δt
                model_points: torch.Tensor,   # (B,M,3)
                target: torch.Tensor,         # (B,M,3)  GT 变换后的点
                idx: torch.Tensor,            # (B,)
                R_prev: torch.Tensor = None,  # (B,3,3)  上一次姿态（可选）
                t_prev: torch.Tensor = None   # (B,3)
                ):
        B, M, _ = model_points.shape
        assert target.shape[:2] == (B, M)
        assert pred_r.shape == (B, 4)
        assert pred_t.shape == (B, 3)

        # 1) 得到“最终姿态” R,t
        if (R_prev is not None) and (t_prev is not None):
            # 增量模式：按上一次姿态对齐后再度量（推荐）
            R, t = self._compose_with_prev(pred_r, pred_t, R_prev, t_prev)
        else:
            # 绝对模式：把 pred_r/pred_t 当作最终姿态
            R = _quat_to_mat(pred_r)                   # (B,3,3)
            t = pred_t                                 # (B,3)

        # 2) 应用姿态到模型点
        # pred = R * X + t
        pred = torch.bmm(model_points, R.transpose(1, 2)) + t.unsqueeze(1)  # (B,M,3)

        # 3) 计算 ADD 或 ADD-S
        # 非对称：逐点 L2 的均值
        # 对称：对每个 pred_i 在 target 上找最近点（用 pairwise 距离取最小）
        dists = []
        for b in range(B):
            if self._is_symmetric(int(idx[b].item())):
                # ADD-S: min over target
                # pairwise: (M,M) = ||pred_b[:,None,:] - target_b[None,:,:]||
                pw = torch.cdist(pred[b].unsqueeze(0), target[b].unsqueeze(0), p=2).squeeze(0)  # (M,M)
                # 对每个 pred 点取最小距离，再对 M 求均值
                d = pw.min(dim=1)[0].mean()
            else:
                # ADD: 对应点逐点距离再均值
                d = (pred[b] - target[b]).norm(dim=1).mean()
            dists.append(d)
        dists = torch.stack(dists, dim=0)  # (B,)

        # 4) reduction
        if self.reduction == "mean":
            loss = dists.mean()
        elif self.reduction == "sum":
            loss = dists.sum()
        else:
            loss = dists  # "none"

        # 返回 (loss, 每样本距离) 便于日志统计
        return loss, dists
