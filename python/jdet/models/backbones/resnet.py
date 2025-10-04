# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved.
# Maintainers:
#     Guowei Yang <471184555@qq.com>
#     Wenyang Zhou <576825820@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""PyTorch implementations of ResNet style backbones.

The original project provides Jittor implementations which expose an
``execute`` method.  For compatibility with the existing code base we
implement the models using ``torch.nn.Module`` while still providing an
``execute`` alias that simply dispatches to ``forward``.
"""

from typing import Callable, Dict, Iterable, List, Optional, Sequence

import torch
from torch import nn
from torch.hub import load_state_dict_from_url

from jdet.utils.registry import BACKBONES

__all__ = [
    "ResNet",
    "Resnet18",
    "Resnet34",
    "Resnet26",
    "Resnet38",
    "Resnet50",
    "Resnet101",
    "Resnet152",
    "Resnext50_32x4d",
    "Resnext101_32x8d",
    "Wide_resnet50_2",
    "Wide_resnet101_2",
    "Resnet18_v1d",
    "Resnet34_v1d",
    "Resnet50_v1d",
    "Resnet101_v1d",
    "Resnet152_v1d",
]


# URL mapping copied from torchvision's ResNet implementation.
_MODEL_URLS: Dict[str, str] = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-cd907fc2.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth",
}


def _conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    # JDet historically used ``execute`` as the forward entry point.
    execute = forward


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = _conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = _conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = _conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    execute = forward


class _ResNetBase(nn.Module):
    """Shared utilities for ResNet style modules."""

    def _freeze_module(self, module: nn.Module) -> None:
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)

    def train(self, mode: bool = True) -> "_ResNetBase":  # type: ignore[override]
        super().train(mode)
        if mode:
            self._freeze_stages()
            if getattr(self, "norm_eval", False):
                for m in self.modules():
                    if isinstance(m, nn.modules.batchnorm._BatchNorm):
                        m.eval()
        return self

    def _freeze_stages(self) -> None:
        raise NotImplementedError


class ResNet(_ResNetBase):
    def __init__(
        self,
        block: Callable[..., nn.Module],
        layers: Sequence[int],
        return_stages: Optional[Iterable[str]] = None,
        frozen_stages: int = -1,
        norm_eval: bool = True,
        num_classes: Optional[int] = None,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[Sequence[bool]] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = (False, False, False)
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be a 3-element tuple"
            )
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )
        self.num_classes = num_classes
        self.return_stages = set(return_stages or ["layer4"])
        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._freeze_stages()

    def _make_layer(
        self,
        block: Callable[..., nn.Module],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            self._freeze_module(self.conv1)
            self._freeze_module(self.bn1)
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f"layer{i}")
            self._freeze_module(m)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i in range(1, 5):
            layer = getattr(self, f"layer{i}")
            x = layer(x)
            name = f"layer{i}"
            if name in self.return_stages:
                outputs.append(x)

        if self.num_classes is not None:
            pooled = self.avgpool(x)
            pooled = torch.flatten(pooled, 1)
            logits = self.fc(pooled)
            if "fc" in self.return_stages:
                outputs.append(logits)

        return outputs

    execute = forward


class ResNet_v1d(_ResNetBase):
    def __init__(
        self,
        block: Callable[..., nn.Module],
        layers: Sequence[int],
        return_stages: Optional[Iterable[str]] = None,
        frozen_stages: int = -1,
        norm_eval: bool = True,
        num_classes: Optional[int] = None,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[Sequence[bool]] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = (False, False, False)
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be a 3-element tuple"
            )
        self.groups = groups
        self.base_width = width_per_group

        self.C1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )
        self.num_classes = num_classes
        self.return_stages = set(return_stages or ["layer4"])
        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._freeze_stages()

    def _make_layer(
        self,
        block: Callable[..., nn.Module],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=False),
                _conv1x1(self.inplanes, planes * block.expansion, 1),
                norm_layer(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            self._freeze_module(self.C1)
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f"layer{i}")
            self._freeze_module(m)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        x = self.C1(x)
        x = self.maxpool(x)

        for i in range(1, 5):
            layer = getattr(self, f"layer{i}")
            x = layer(x)
            name = f"layer{i}"
            if name in self.return_stages:
                outputs.append(x)

        if self.num_classes is not None:
            pooled = self.avgpool(x)
            pooled = torch.flatten(pooled, 1)
            logits = self.fc(pooled)
            if "fc" in self.return_stages:
                outputs.append(logits)

        return outputs

    execute = forward


def _load_pretrained(model: nn.Module, arch: str) -> nn.Module:
    url = _MODEL_URLS.get(arch)
    if url is None:
        raise ValueError(f"No pretrained weights available for architecture: {arch}")
    try:
        state_dict = load_state_dict_from_url(url, progress=True)
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"Failed to download pretrained weights for {arch}. "
            "Check your network connection."
        ) from exc
    model.load_state_dict(state_dict)
    return model


def _resnet(block: Callable[..., nn.Module], layers: Sequence[int], **kwargs) -> ResNet:
    return ResNet(block, layers, **kwargs)


@BACKBONES.register_module()
def Resnet18(pretrained: bool = False, **kwargs) -> ResNet:
    model = _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        _load_pretrained(model, "resnet18")
    return model


# Alias used elsewhere in the project.
resnet18 = Resnet18


@BACKBONES.register_module()
def Resnet34(pretrained: bool = False, **kwargs) -> ResNet:
    model = _resnet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        _load_pretrained(model, "resnet34")
    return model


resnet34 = Resnet34


@BACKBONES.register_module()
def Resnet50(pretrained: bool = False, **kwargs) -> ResNet:
    model = _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        _load_pretrained(model, "resnet50")
    return model


@BACKBONES.register_module()
def Resnet38(**kwargs) -> ResNet:
    return _resnet(Bottleneck, [2, 3, 5, 2], **kwargs)


@BACKBONES.register_module()
def Resnet26(**kwargs) -> ResNet:
    return _resnet(Bottleneck, [1, 2, 4, 1], **kwargs)


@BACKBONES.register_module()
def Resnet101(pretrained: bool = False, **kwargs) -> ResNet:
    model = _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        _load_pretrained(model, "resnet101")
    return model


@BACKBONES.register_module()
def Resnet152(pretrained: bool = False, **kwargs) -> ResNet:
    model = _resnet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        _load_pretrained(model, "resnet152")
    return model


@BACKBONES.register_module()
def Resnext50_32x4d(pretrained: bool = False, **kwargs) -> ResNet:
    kwargs.setdefault("groups", 32)
    kwargs.setdefault("width_per_group", 4)
    model = _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        _load_pretrained(model, "resnext50_32x4d")
    return model


@BACKBONES.register_module()
def Resnext101_32x8d(pretrained: bool = False, **kwargs) -> ResNet:
    kwargs.setdefault("groups", 32)
    kwargs.setdefault("width_per_group", 8)
    model = _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        _load_pretrained(model, "resnext101_32x8d")
    return model


@BACKBONES.register_module()
def Wide_resnet50_2(pretrained: bool = False, **kwargs) -> ResNet:
    kwargs.setdefault("width_per_group", 64 * 2)
    model = _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        _load_pretrained(model, "wide_resnet50_2")
    return model


@BACKBONES.register_module()
def Wide_resnet101_2(pretrained: bool = False, **kwargs) -> ResNet:
    kwargs.setdefault("width_per_group", 64 * 2)
    model = _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        _load_pretrained(model, "wide_resnet101_2")
    return model


class _ResNetV1d(ResNet_v1d):
    pass


def _resnet_v1d(block: Callable[..., nn.Module], layers: Sequence[int], **kwargs) -> ResNet_v1d:
    return ResNet_v1d(block, layers, **kwargs)


@BACKBONES.register_module()
def Resnet18_v1d(pretrained: bool = False, **kwargs) -> ResNet_v1d:
    model = _resnet_v1d(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        _load_pretrained(model, "resnet18")
    return model


@BACKBONES.register_module()
def Resnet34_v1d(pretrained: bool = False, **kwargs) -> ResNet_v1d:
    model = _resnet_v1d(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        _load_pretrained(model, "resnet34")
    return model


@BACKBONES.register_module()
def Resnet50_v1d(pretrained: bool = False, **kwargs) -> ResNet_v1d:
    model = _resnet_v1d(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        _load_pretrained(model, "resnet50")
    return model


@BACKBONES.register_module()
def Resnet101_v1d(pretrained: bool = False, **kwargs) -> ResNet_v1d:
    model = _resnet_v1d(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        _load_pretrained(model, "resnet101")
    return model


@BACKBONES.register_module()
def Resnet152_v1d(pretrained: bool = False, **kwargs) -> ResNet_v1d:
    model = _resnet_v1d(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        _load_pretrained(model, "resnet152")
    return model
