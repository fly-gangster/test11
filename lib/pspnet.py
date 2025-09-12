# lib/pspnet.py
import torch
from torch import nn
from torch.nn import functional as F
import lib.extractors as extractors


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [
            F.interpolate(stage(feats), size=(h, w), mode='bilinear', align_corners=True)
            for stage in self.stages
        ] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)


class PSPNet(nn.Module):
    """
    backend: {'resnet18','resnet34','resnet50','resnet101','resnet152'}
    forward 返回 logits（对每个像素的类别分数，已 LogSoftmax），形状 (B, n_classes, H, W)
    """
    def __init__(self, n_classes=21, sizes=(1, 2, 3, 6),
                 psp_size=None, deep_features_size=None,
                 backend='resnet18', pretrained=False):
        super().__init__()
        self.feats = getattr(extractors, backend)(pretrained=pretrained)
        # 自动根据 backbone 决定通道数：18/34 -> 512；50/101/152 -> 2048
        if backend in ('resnet18', 'resnet34'):
            _in_ch = 512
            _clf_ch = 512
        else:
            _in_ch = 2048
            _clf_ch = 2048
        psp_size = _in_ch if psp_size is None else psp_size
        deep_features_size = _clf_ch if deep_features_size is None else deep_features_size

        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        # 最后一层通道数应等于 n_classes
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1, bias=True),
            nn.LogSoftmax(dim=1)
        )

        # 可选的图像级分类分支（若需要）
        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_classes)
        )

    # lib/pspnet.py
    def forward(self, x, return_features: bool = False):
        f, class_f = self.feats(x)
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p);
        p = self.drop_2(p)
        p = self.up_2(p);
        p = self.drop_2(p)
        p = self.up_3(p)  # <-- (B, 64, H', W')

        logits = self.final(p)  # (B, n_classes, H', W')
        if return_features:
            return logits, p
        return logits


