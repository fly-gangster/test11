"""
Reworked DenseFusion training script with
- --num_points command‑line arg (default 4)
- parameter propagated to Dataset / Network / Loss
- minimal refactor; other logic identical to original
"""

import _init_paths  # noqa: F401
import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets.ycb.dataset import PoseDataset as PoseDataset_ycb
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.utils import setup_logger

# ---------------------- argparse ----------------------
parser = argparse.ArgumentParser(description="DenseFusion Training")
parser.add_argument('--dataset', type=str, default='ycb', choices=['ycb', 'linemod'])
parser.add_argument('--dataset_root', type=str, required=True,
                    help="path to YCB_Video_Dataset or Linemod_preprocessed")
parser.add_argument('--num_points', type=int, default=4, help='sampled points per frame')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_rate', type=float, default=0.3)
parser.add_argument('--w', type=float, default=0.015)
parser.add_argument('--w_rate', type=float, default=0.3)
parser.add_argument('--decay_margin', type=float, default=0.016)
parser.add_argument('--refine_margin', type=float, default=0.013)
parser.add_argument('--noise_trans', type=float, default=0.03)
parser.add_argument('--iteration', type=int, default=2)
parser.add_argument('--nepoch', type=int, default=500)
parser.add_argument('--resume_posenet', type=str, default='')
parser.add_argument('--resume_refinenet', type=str, default='')
parser.add_argument('--start_epoch', type=int, default=1)
opt = parser.parse_args()

# ---------------- reproducibility ----------------
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
cudnn.benchmark = True

# --------------- dataset‑specific params ---------------
if opt.dataset == 'ycb':
    opt.num_objects = 21
    opt.outf = 'trained_models/ycb'
    opt.log_dir = 'experiments/logs/ycb'
    opt.repeat_epoch = 1
elif opt.dataset == 'linemod':
    opt.num_objects = 13
    opt.outf = 'trained_models/linemod'
    opt.log_dir = 'experiments/logs/linemod'
    opt.repeat_epoch = 20
else:
    raise ValueError('Unknown dataset')

# ensure dirs
Path(opt.outf).mkdir(parents=True, exist_ok=True)
Path(opt.log_dir).mkdir(parents=True, exist_ok=True)

# ---------------- network ----------------
estimator = PoseNet(num_points=opt.num_points, num_obj=opt.num_objects).cuda()
refiner = PoseRefineNet(num_points=opt.num_points, num_obj=opt.num_objects).cuda()

if opt.resume_posenet:
    estimator.load_state_dict(torch.load(os.path.join(opt.outf, opt.resume_posenet)))
if opt.resume_refinenet:
    refiner.load_state_dict(torch.load(os.path.join(opt.outf, opt.resume_refinenet)))

# ---------------- optimizer ----------------
opt.refine_start = bool(opt.resume_refinenet)
optimizer = optim.Adam((refiner if opt.refine_start else estimator).parameters(), lr=opt.lr)

# ---------------- dataset loader ----------------
DatasetCls = PoseDataset_ycb if opt.dataset == 'ycb' else PoseDataset_linemod
train_set = DatasetCls('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=opt.workers)

test_set = DatasetCls('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=opt.workers)

opt.sym_list = train_set.get_sym_list()
opt.num_points_mesh = train_set.get_num_points_mesh()
print(f"Dataset loaded: train={len(train_set)}, test={len(test_set)}, pts_mesh={opt.num_points_mesh}")

criterion = Loss(opt.num_points_mesh, opt.sym_list)
criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)

best_test = float('inf')
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=opt.lr_rate, patience=5)

# ---------------- training loop ----------------
start_time = time.time()
for epoch in range(opt.start_epoch, opt.nepoch + 1):
    logger = setup_logger(f'epoch{epoch}', os.path.join(opt.log_dir, f'epoch_{epoch}.log'))
    logger.info(f"Epoch {epoch} starting, elapsed {time.time()-start_time:.1f}s, num_points={opt.num_points}")
    estimator.train(); refiner.train() if opt.refine_start else None

    train_count = 0; train_dis_sum = 0.0
    optimizer.zero_grad()

    for rep in range(opt.repeat_epoch):
        for data in train_loader:
            points, choose, img, target, model_points, idx = [d.cuda() for d in data]
            pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
            loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target,
                                                           model_points, idx, points, opt.w, opt.refine_start)
            if opt.refine_start:
                for _ in range(opt.iteration):
                    pr, pt = refiner(new_points, emb, idx)
                    dis, new_points, new_target = criterion_refine(pr, pt, new_target, model_points, idx, new_points)
                    dis.backward()
            else:
                loss.backward()

            train_dis_sum += dis.item(); train_count += 1
            if train_count % opt.batch_size == 0:
                logger.info(f'Train Epoch {epoch} Batch {train_count//opt.batch_size} dis={train_dis_sum/opt.batch_size:.4f}')
                optimizer.step(); optimizer.zero_grad(); train_dis_sum = 0

    # ---------------- validation ----------------
estimator.eval(); refiner.eval()
    with torch.no_grad():
        test_dis_sum = 0; test_cnt = 0
        for data in test_loader:
            points, choose, img, target, model_points, idx = [d.cuda() for d in data]
            pr, pt, pc, emb = estimator(img, points, choose, idx)
            _, dis, np_new, nt_new = criterion(pr, pt, pc, target, model_points, idx, points, opt.w, opt.refine_start)
            if opt.refine_start:
                for _ in range(opt.iteration):
                    pr, pt = refiner(np_new, emb, idx)
                    dis, np_new, nt_new = criterion_refine(pr, pt, nt_new, model_points, idx, np_new)
            test_dis_sum += dis.item(); test_cnt += 1
        test_dis = test_dis_sum / test_cnt
        logger.info(f'Epoch {epoch} TEST Avg dis={test_dis:.4f}')

    # ---------------- scheduler / save ----------------
    scheduler.step(test_dis)
    torch.save(estimator.state_dict(), os.path.join(opt.outf, 'posenet_latest.pth'))
    if test_dis < best_test:
        best_test = test_dis
        torch.save(estimator.state_dict(), os.path.join(opt.outf, f'posenet_best_{best_test:.4f}.pth'))
        logger.info('>>> Best model saved')

    # start refine if条件满足
    if best_test < opt.refine_margin and not opt.refine_start:
        opt.refine_start = True
        optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)
        logger.info('>>> Switch to Refine stage')

if __name__ == '__main__':
    pass  # main() is executed upon import above
