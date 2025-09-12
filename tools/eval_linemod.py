# tools/eval_linemod.py
import argparse
import os
import sys
import copy
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# --- 加入工程根目录到 sys.path，避免 import 找不到 ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "lib")):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---- imports: 兼容两种包路径 ----
try:
    from lib.network import PoseNet, PoseRefineNet
except Exception:
    from network import PoseNet, PoseRefineNet

try:
    from lib.transformations import quaternion_matrix, quaternion_from_matrix
except Exception:
    from transformations import quaternion_matrix, quaternion_from_matrix

# 你的数据集实现保持原接口：
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod


def load_state(m, path, name="model"):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    try:
        m.load_state_dict(sd, strict=True)
        print(f"[OK] Loaded {name} weights strictly: {path}")
    except Exception as e:
        print(f"[WARN] Strict load failed for {name}: {e}\n       -> retry with strict=False")
        m.load_state_dict(sd, strict=False)
        print(f"[OK] Loaded {name} with strict=False: {path}")


def build_diameters(models_info_yml, objlist, add_fraction=0.1):
    with open(models_info_yml, "r") as f:
        meta = yaml.safe_load(f)
    diameter = []
    for obj_id in objlist:
        d_raw = float(meta[obj_id]["diameter"])
        # 经验：>10 基本是 mm，<=10 基本已是 m
        d_m = d_raw / 1000.0 if d_raw > 10.0 else d_raw
        diameter.append(d_m * add_fraction)
    return diameter


def add_or_adds(pred_xyz, target_xyz, symmetric=False):
    """
    pred_xyz: (P,3)  预测后的模型点
    target_xyz: (P,3) GT 对齐的模型点
    return: 标量距离（平均点距）
    """
    pred = torch.as_tensor(pred_xyz, dtype=torch.float32, device="cuda")
    tgt  = torch.as_tensor(target_xyz, dtype=torch.float32, device="cuda")
    if symmetric:
        # ADD-S: 每个 pred 点找 target 最近邻（更稳，不依赖外部 KNN）
        dmat = torch.cdist(pred[None, ...], tgt[None, ...]).squeeze(0)  # (P,P)
        return torch.min(dmat, dim=1)[0].mean().item()
    else:
        # ADD: 一一对应
        return torch.norm(pred - tgt, dim=1).mean().item()


def _guess_scale_to_meter(pts_np: np.ndarray) -> float:
    """根据点云尺度猜测是否为 mm；>1（常见几十~几百）判为 mm→返回 0.001，否则返回 1.0"""
    if pts_np.size == 0:
        return 1.0
    med = float(np.median(np.linalg.norm(pts_np, axis=1)))
    return 0.001 if med > 1.0 else 1.0


def map_class_index(idx_raw: int, objlist: list) -> int:
    """把原始的 idx（可能是 0..12 或实际物体ID）映射到 0..len(objlist)-1；异常返回 -1"""
    if idx_raw in objlist:
        return objlist.index(idx_raw)
    if 0 <= idx_raw < len(objlist):
        return idx_raw
    return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="dataset root dir (Linemod_preprocessed)")
    parser.add_argument("--model", type=str, required=True,
                        help="PoseNet weight path (.pth)")
    parser.add_argument("--refine_model", type=str, default="",
                        help="PoseRefineNet weight path (.pth); empty to disable")
    parser.add_argument("--dataset_config_dir", type=str, default="datasets/linemod/dataset_config",
                        help="dir that contains models_info.yml")
    parser.add_argument("--output_result_dir", type=str, default="experiments/eval_result/linemod",
                        help="where eval logs go")
    parser.add_argument("--num_points", type=int, default=500)
    parser.add_argument("--iteration", type=int, default=4,
                        help="refine iterations (ignored if no refine_model)")
    parser.add_argument("--add_fraction", type=float, default=0.1,
                        help="ADD threshold = fraction * object diameter")
    parser.add_argument("--units", choices=["auto", "m", "mm"], default="auto",
                        help="model_points/target 的单位；auto 会猜测 mm→m")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_result_dir, exist_ok=True)

    # 固定评测设置（与原 Linemod 一致）
    num_objects = 13
    objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]  # 与数据集 idx 对应表
    bs = 1

    # 随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 直径表（单位安全）
    models_info_yml = os.path.join(args.dataset_config_dir, "models_info.yml")
    diameter = build_diameters(models_info_yml, objlist, args.add_fraction)
    print("[CHECK] 10% diameters (m):", ["%.5f" % d for d in diameter])

    # 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    estimator = PoseNet(num_points=args.num_points, num_obj=num_objects).to(device)
    load_state(estimator, args.model, "PoseNet")
    estimator.eval()

    use_refine = bool(args.refine_model and os.path.isfile(args.refine_model))
    if use_refine:
        refiner = PoseRefineNet(num_points=args.num_points, num_obj=num_objects).to(device)
        load_state(refiner, args.refine_model, "PoseRefineNet")
        refiner.eval()
        iteration = max(1, int(args.iteration))
        print(f"[INFO] Refiner enabled with {iteration} iterations.")
    else:
        refiner = None
        iteration = 0
        print("[INFO] Refiner disabled (no valid --refine_model provided).")

    # 数据
    testdataset = PoseDataset_linemod(
        'eval', args.num_points, False, args.dataset_root, 0.0, True
    )
    testloader = torch.utils.data.DataLoader(
        testdataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )
    sym_list = set(testdataset.get_sym_list())  # 可能是 0..12 或真实物体ID
    num_points_mesh = testdataset.get_num_points_mesh()
    print(f"[INFO] sym_list={sym_list}")

    # 记录
    success_count = [0 for _ in range(num_objects)]
    num_count = [0 for _ in range(num_objects)]
    log_path = os.path.join(args.output_result_dir, "eval_result_logs.txt")
    fw = open(log_path, "w")

    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            points, choose, img, target, model_points, idx = data  # idx 可能是 [0..12] 或物体ID
            if len(points.size()) == 2:
                msg = f"No.{i} NOT Pass! Lost detection!"
                print(msg); fw.write(msg + "\n")
                continue

            points = Variable(points).to(device)            # (B,N,3)
            choose = Variable(choose).to(device)            # (B,1,N) or (B,N)
            img    = Variable(img).to(device)               # (B,3,H,W)
            target = Variable(target).to(device)            # (B,P,3) P=模型点数
            model_points = Variable(model_points).to(device)# (B,P,3)
            idx    = Variable(idx).long().to(device)        # (B,1) / (B,)

            # ---- 初始预测 ----
            pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)   # 见 network.PoseNet.forward
            # 形状：(B,N,4/3/1)，且 r 是逐点预测的四元数
            pred_r = pred_r / (torch.norm(pred_r, dim=2, keepdim=True) + 1e-8)  # 归一化
            pred_c = pred_c.view(bs, args.num_points)                            # (B,N)
            which_max = torch.argmax(pred_c, dim=1)                              # (B,)
            # 取置信度最高的点的 r,t
            r0 = pred_r[0, which_max[0]]     # (4,)
            t0 = (points[0] + pred_t[0])[which_max[0]]  # (3,)
            my_r = r0.detach().cpu().numpy()
            my_t = t0.detach().cpu().numpy()

            # ---- 细化迭代 ----
            for _ in range(iteration):
                R_np = quaternion_matrix(my_r).astype(np.float32)[:3, :3]    # (3,3)
                T_np = my_t.astype(np.float32)                               # (3,)

                R = torch.from_numpy(R_np).to(device).view(1, 3, 3)          # (1,3,3)
                T = torch.from_numpy(T_np).to(device).view(1, 1, 3).repeat(1, args.num_points, 1)  # (1,N,3)

                new_points = torch.bmm(points - T, R).contiguous()           # (1,N,3)
                rx2, tx2 = refiner(new_points, emb, idx) if refiner is not None else (None, None)

                if rx2 is None:
                    break

                r2 = rx2[0].detach().cpu().numpy()
                t2 = tx2[0].detach().cpu().numpy()

                M1 = quaternion_matrix(my_r); M1[0:3, 3] = my_t
                M2 = quaternion_matrix(r2);  M2[0:3, 3] = t2
                M  = np.dot(M1, M2)

                R_final = copy.deepcopy(M); R_final[0:3, 3] = 0
                my_r = quaternion_from_matrix(R_final, True)
                my_t = np.array([M[0, 3], M[1, 3], M[2, 3]], dtype=np.float32)

            # ---- 计算 ADD / ADD-S（统一单位为米） ----
            mp = model_points[0].detach().cpu().numpy()  # (P,3)
            gt = target[0].detach().cpu().numpy()        # (P,3)

            if args.units == "m":
                s_pts = 1.0
            elif args.units == "mm":
                s_pts = 0.001
            else:
                s_pts = min(_guess_scale_to_meter(mp), _guess_scale_to_meter(gt))
            if i == 0:
                print(f"[UNIT] scale for points/GT = {s_pts}  (1.0=m, 0.001=mm→m)")

            mp = mp * s_pts
            gt = gt * s_pts
            R_pred = quaternion_matrix(my_r)[:3, :3]     # (3,3)
            pred_xyz = np.dot(mp, R_pred.T) + (my_t * s_pts)  # (P,3)

            # ---- 类别映射（鲁棒） ----
            cls_raw = int(idx[0].item())
            cls = map_class_index(cls_raw, objlist)
            if cls < 0:
                print(f"[WARN] unexpected class index {cls_raw}; skip")
                fw.write(f"[WARN] unexpected class index {cls_raw}; skip\n")
                continue

            # 对称判断：兼容 sym_list 存 0..12 或实际物体ID 的两种情况
            is_sym = (cls_raw in sym_list) or (objlist[cls] in sym_list)

            dis = add_or_adds(pred_xyz, gt, symmetric=is_sym)
            thr = diameter[cls]
            ratio = dis / thr if thr > 0 else float("inf")

            # === 你要的调试打印（放在判定前） ===
            if i < 10:
                print(f"[DBG] dis={dis:.6f} m, thr={thr:.6f} m, ratio={ratio:.3f}, cls_raw={cls_raw}→cls={cls}")

            # 用 cls/thr 做通过判定与计数
            if dis < thr:
                success_count[cls] += 1
                msg = f"No.{i} Pass! Distance: {dis:.6f}"
            else:
                msg = f"No.{i} NOT Pass! Distance: {dis:.6f}"
            print(msg); fw.write(msg + "\n")
            num_count[cls] += 1

    # ---- 汇总 ----
    for k in range(num_objects):
        rate = 0.0 if num_count[k] == 0 else float(success_count[k]) / num_count[k]
        print(f"Object {objlist[k]} success rate: {rate:.6f}")
        fw.write(f"Object {objlist[k]} success rate: {rate:.6f}\n")
    all_rate = 0.0 if sum(num_count) == 0 else float(sum(success_count)) / sum(num_count)
    print(f"ALL success rate: {all_rate:.6f}")
    fw.write(f"ALL success rate: {all_rate:.6f}\n")
    fw.close()
    print(f"[DONE] Log saved to: {log_path}")


if __name__ == "__main__":
    main()
