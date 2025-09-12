import os, random, copy
import numpy as np
import numpy.ma as ma
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.utils.data as data
from torchvision import transforms
import scipy.io as scio


class SegDataset(data.Dataset):
    """
    语义分割训练集（YCB-Segmentation）。
    参数
    ----
    root_dir     : 数据集根目录
    txtlist      : 帧列表 txt
    use_noise    : 是否做颜色/几何增强
    dataset_len  : __len__ 返回的 epoch 长度（> 实际帧数时会随机重复）
    """
    def __init__(self, root_dir, txtlist, use_noise=True, dataset_len=5000):
        self.root = root_dir
        self.use_noise = use_noise
        self.path, self.real_path = [], []

        with open(txtlist, 'r') as f:
            for line in f:
                frame = line.strip()
                self.path.append(frame)
                if frame.startswith('data/'):
                    self.real_path.append(frame)

        self.data_len = len(self.path)
        self.back_len = len(self.real_path)
        self.length = dataset_len

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        self.back_front = np.ones((480, 640), dtype=np.uint8)

    # ------------------------------------------------------------ #
    def __getitem__(self, idx):
        # 随机选一帧，确保 dataset_len > 实际帧数 时也能遍历
        index = random.randint(0, self.data_len - 1)

        rgb_path   = f'{self.root}/{self.path[index]}-color.png'
        label_path = f'{self.root}/{self.path[index]}-label.png'
        meta_path  = f'{self.root}/{self.path[index]}-meta.mat'

        rgb   = Image.open(rgb_path).convert('RGB')
        label = np.array(Image.open(label_path))
        meta  = scio.loadmat(meta_path)
        obj_ids = np.append(meta['cls_indexes'].flatten().astype(np.int32), [0])

        # ------------ 数据增强 ------------ #
        if self.use_noise:
            rgb = self.trancolor(rgb)

        if self.path[index].startswith('data_syn'):
            rgb = self._blend_synthetic(rgb, label)

        # 随机翻转
        if self.use_noise:
            choice = random.randint(0, 3)
            if choice & 1:  # 左右翻
                rgb, label = rgb.transpose(Image.FLIP_LEFT_RIGHT), np.fliplr(label)
            if choice & 2:  # 上下翻
                rgb, label = rgb.transpose(Image.FLIP_TOP_BOTTOM), np.flipud(label)

        # ------------ to Tensor ------------ #
        rgb = self.norm(torch.from_numpy(np.transpose(np.array(rgb), (2, 0, 1)).astype(np.float32)))
        target = torch.from_numpy(label.astype(np.int64))

        return rgb, target

    def __len__(self):
        return self.length

    # ------------------------------------------------------------ #
    def _blend_synthetic(self, rgb_img, label):
        """把合成帧随机叠在 real 背景上，增加真实感。"""
        rgb = np.array(ImageEnhance.Brightness(rgb_img).enhance(1.5)
                       .filter(ImageFilter.GaussianBlur(radius=0.8)))
        seed = random.randint(0, self.back_len - 1)
        back_rgb = Image.open(f'{self.root}/{self.path[seed]}-color.png').convert('RGB')
        back_rgb = np.array(self.trancolor(back_rgb))
        back_label = np.array(Image.open(f'{self.root}/{self.path[seed]}-label.png'))

        mask = ma.getmaskarray(ma.masked_equal(label, 0))
        rgb = np.transpose(rgb + np.random.normal(0, 5.0, rgb.shape), (2, 0, 1))
        back_rgb = np.transpose(back_rgb, (2, 0, 1))
        blended = back_rgb * mask + rgb
        blended = np.transpose(blended, (1, 2, 0))
        new_label = back_label * mask + label
        return Image.fromarray(blended.astype(np.uint8))
