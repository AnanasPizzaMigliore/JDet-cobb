"""Reusable building blocks implemented with PyTorch."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import warnings

import torch
from torch import nn

from jdet.utils.registry import BRICKS, build_from_cfg

from .weight_init import kaiming_init


# Padding operators
BRICKS.register_module("zero", module=nn.ZeroPad2d)
BRICKS.register_module("reflect", module=nn.ReflectionPad2d)
BRICKS.register_module("replicate", module=nn.ReplicationPad2d)


# Convolutions
BRICKS.register_module("Conv1d", module=nn.Conv1d)
BRICKS.register_module("Conv2d", module=nn.Conv2d)
BRICKS.register_module("Conv3d", module=nn.Conv3d)
BRICKS.register_module("Conv", module=nn.Conv2d)


class _BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features: Optional[int] = None, in_channels: Optional[int] = None, **kwargs):
        if num_features is None:
            if in_channels is None:
                raise TypeError("BatchNorm2d requires num_features or in_channels")
            num_features = in_channels
        super().__init__(num_features, **kwargs)


class _BatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features: Optional[int] = None, in_channels: Optional[int] = None, **kwargs):
        if num_features is None:
            if in_channels is None:
                raise TypeError("BatchNorm1d requires num_features or in_channels")
            num_features = in_channels
        super().__init__(num_features, **kwargs)


class _BatchNorm3d(nn.BatchNorm3d):
    def __init__(self, num_features: Optional[int] = None, in_channels: Optional[int] = None, **kwargs):
        if num_features is None:
            if in_channels is None:
                raise TypeError("BatchNorm3d requires num_features or in_channels")
            num_features = in_channels
        super().__init__(num_features, **kwargs)


BRICKS.register_module("BN", module=_BatchNorm2d)
BRICKS.register_module("BN1d", module=_BatchNorm1d)
BRICKS.register_module("BN2d", module=_BatchNorm2d)
BRICKS.register_module("BN3d", module=_BatchNorm3d)


class _InstanceNorm(nn.InstanceNorm2d):
    def __init__(self, num_features: Optional[int] = None, in_channels: Optional[int] = None, **kwargs):
        if num_features is None:
            if in_channels is None:
                raise TypeError("InstanceNorm requires num_features or in_channels")
            num_features = in_channels
        super().__init__(num_features, **kwargs)


class _InstanceNorm1d(nn.InstanceNorm1d):
    def __init__(self, num_features: Optional[int] = None, in_channels: Optional[int] = None, **kwargs):
        if num_features is None:
            if in_channels is None:
                raise TypeError("InstanceNorm1d requires num_features or in_channels")
            num_features = in_channels
        super().__init__(num_features, **kwargs)


class _InstanceNorm3d(nn.InstanceNorm3d):
    def __init__(self, num_features: Optional[int] = None, in_channels: Optional[int] = None, **kwargs):
        if num_features is None:
            if in_channels is None:
                raise TypeError("InstanceNorm3d requires num_features or in_channels")
            num_features = in_channels
        super().__init__(num_features, **kwargs)


BRICKS.register_module("IN", module=_InstanceNorm)
BRICKS.register_module("IN1d", module=_InstanceNorm1d)
BRICKS.register_module("IN2d", module=_InstanceNorm)
BRICKS.register_module("IN3d", module=_InstanceNorm3d)


class _GroupNorm(nn.GroupNorm):
    def __init__(self, num_groups: int, num_channels: Optional[int] = None, in_channels: Optional[int] = None, **kwargs):
        if num_channels is None:
            if in_channels is None:
                raise TypeError("GroupNorm requires num_channels or in_channels")
            num_channels = in_channels
        super().__init__(num_groups, num_channels, **kwargs)


BRICKS.register_module("GN", module=_GroupNorm)
BRICKS.register_module("LN", module=nn.LayerNorm)


# Activations
BRICKS.register_module("ReLU", module=nn.ReLU)
BRICKS.register_module("LeakyReLU", module=nn.LeakyReLU)
BRICKS.register_module("PReLU", module=nn.PReLU)
BRICKS.register_module("ReLU6", module=nn.ReLU6)
BRICKS.register_module("ELU", module=nn.ELU)
BRICKS.register_module("Sigmoid", module=nn.Sigmoid)
BRICKS.register_module("Tanh", module=nn.Tanh)
BRICKS.register_module("GELU", module=nn.GELU)


class ConvModule(nn.Module):
    """A convolution + normalization + activation block."""

    _abbr_ = "conv_block"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] | int,
        stride: Tuple[int, int] | int = 1,
        padding: Tuple[int, int] | int = 0,
        dilation: Tuple[int, int] | int = 1,
        groups: int = 1,
        bias: bool | str = "auto",
        conv_cfg: Optional[Dict] = None,
        norm_cfg: Optional[Dict] = None,
        act_cfg: Optional[Dict] = dict(type="ReLU"),
        with_spectral_norm: bool = False,
        padding_mode: str = "zeros",
        order: Tuple[str, str, str] = ("conv", "norm", "act"),
    ) -> None:
        super().__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        assert set(order) == {"conv", "norm", "act"}

        official_padding_mode = {"zeros", "circular"}
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None

        if bias == "auto":
            bias = not self.with_norm
        if self.with_norm and bias:
            warnings.warn("ConvModule has both norm and bias enabled")

        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.padding = padding
        self.order = order

        if self.with_explicit_padding:
            pad_cfg = dict(type=padding_mode)
            self.padding_layer = build_from_cfg(pad_cfg, BRICKS, padding=padding)
            conv_padding = 0
        else:
            self.padding_layer = None
            conv_padding = padding

        conv_cfg = dict(type="Conv2d") if conv_cfg is None else conv_cfg.copy()
        self.conv = build_from_cfg(
            conv_cfg,
            BRICKS,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        if with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        if self.with_norm:
            norm_cfg = norm_cfg.copy()
            norm_type = norm_cfg.pop("type")
            if norm_type in {"GN"}:
                self.norm = build_from_cfg(
                    dict(type=norm_type, **norm_cfg),
                    BRICKS,
                    num_channels=out_channels if order.index("norm") > order.index("conv") else in_channels,
                )
            elif norm_type in {"LN"}:
                normalized_shape = out_channels if order.index("norm") > order.index("conv") else in_channels
                self.norm = build_from_cfg(dict(type=norm_type, **norm_cfg), BRICKS, normalized_shape=normalized_shape)
            else:
                num_features = out_channels if order.index("norm") > order.index("conv") else in_channels
                self.norm = build_from_cfg(
                    dict(type=norm_type, **norm_cfg),
                    BRICKS,
                    num_features=num_features,
                    in_channels=num_features,
                )
        else:
            self.norm = None

        if self.with_activation:
            act_cfg = act_cfg.copy()
            self.activate = build_from_cfg(act_cfg, BRICKS)
        else:
            self.activate = None

        self.init_weights()

    def init_weights(self) -> None:
        kaiming_init(self.conv, mode="fan_out")
        if self.with_norm:
            if isinstance(self.norm, nn.BatchNorm2d):
                nn.init.constant_(self.norm.weight, 1)
                nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.order:
            if layer == "conv":
                if self.with_explicit_padding and self.padding_layer is not None:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == "norm" and self.with_norm:
                x = self.norm(x)
            elif layer == "act" and self.with_activation:
                x = self.activate(x)
        return x


    def execute(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - legacy API
        return self.forward(x)
