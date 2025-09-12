# lib/extractors.py
from collections import OrderedDict
import math
import torch
import torch.nn as nn

try:
    import torchvision.models as tvm
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False


def load_weights_sequential(target: nn.Module, source_state: OrderedDict, strict: bool = False):
    """将 source_state 的参数按键名尽可能加载到 target；默认 strict=False 更健壮。"""
    target.load_state_dict(source_state, strict=strict)


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, dilation=dilation, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation,
            padding=dilation, bias=False
        )
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * 4)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    与 torchvision 的命名基本一致，但 layer3/layer4 支持空洞卷积（dilation）。
    forward 返回 (x, x_3)，与原仓库 PSPNet 对接保持一致：
      - x   : layer4 的输出
      - x_3 : layer3 的输出（用于 deep features）
    """
    def __init__(self, block, layers=(3, 4, 23, 3)):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 与常见语义分割设置一致：layer2 下采样；layer3/4 保持步幅=1，用 dilation 扩大感受野
        self.layer1 = self._make_layer(block,  64, layers[0], stride=1, dilation=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilation=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias,   0.0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample, dilation=dilation)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, downsample=None, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x_3 = self.layer3(x)
        x = self.layer4(x_3)
        return x, x_3


# ---- 构造函数（与原接口保持一致） ----

def _load_torchvision_backbone(name: str):
    if not _HAS_TORCHVISION:
        return None
    # 兼容老/新 API
    fn = getattr(tvm, name, None)
    if fn is None:
        return None
    try:
        # 新版 API
        model = fn(weights='IMAGENET1K_V1')
    except TypeError:
        # 旧版 API
        model = fn(pretrained=True)
    return model


def _load_pretrained_roughly(ours: nn.Module, tv_name: str):
    tv_model = _load_torchvision_backbone(tv_name)
    if tv_model is None:
        return
    sd = tv_model.state_dict()
    # 直接尝试加载；由于结构名基本一致，通常能匹配大部分参数
    ours.load_state_dict(sd, strict=False)


def resnet18(pretrained: bool = False):
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        _load_pretrained_roughly(model, 'resnet18')
    return model

def resnet34(pretrained: bool = False):
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        _load_pretrained_roughly(model, 'resnet34')
    return model

def resnet50(pretrained: bool = False):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        _load_pretrained_roughly(model, 'resnet50')
    return model

def resnet101(pretrained: bool = False):
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    if pretrained:
        _load_pretrained_roughly(model, 'resnet101')
    return model

def resnet152(pretrained: bool = False):
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    if pretrained:
        _load_pretrained_roughly(model, 'resnet152')
    return model