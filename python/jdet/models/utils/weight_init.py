"""Weight initialisation helpers implemented with PyTorch."""

from __future__ import annotations

from typing import Iterable

import math

import torch
from torch import nn
from torch.nn import init as nn_init


def _get_tensor(param):
    return isinstance(param, (torch.Tensor, nn.Parameter))


def normal_init(module: nn.Module, mean: float = 0, std: float = 1, bias: float = 0) -> None:
    if hasattr(module, "weight") and _get_tensor(module.weight):
        nn_init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and _get_tensor(module.bias):
        nn_init.constant_(module.bias, bias)


def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    if hasattr(module, "weight") and _get_tensor(module.weight):
        nn_init.constant_(module.weight, val)
    if hasattr(module, "bias") and _get_tensor(module.bias):
        nn_init.constant_(module.bias, bias)


def xavier_init(module: nn.Module, gain: float = 1, bias: float = 0, distribution: str = "normal") -> None:
    assert distribution in ["uniform", "normal"], "distribution must be 'uniform' or 'normal'"
    if hasattr(module, "weight") and _get_tensor(module.weight):
        if distribution == "uniform":
            nn_init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn_init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and _get_tensor(module.bias):
        nn_init.constant_(module.bias, bias)


def trunc_normal_init(
    module: nn.Module,
    mean: float = 0,
    std: float = 1,
    a: float = -2,
    b: float = 2,
    bias: float = 0,
) -> None:
    if hasattr(module, "weight") and _get_tensor(module.weight):
        nn_init.trunc_normal_(module.weight, mean=mean, std=std, a=a, b=b)
    if hasattr(module, "bias") and _get_tensor(module.bias):
        nn_init.constant_(module.bias, bias)


def uniform_init(module: nn.Module, a: float = 0, b: float = 1, bias: float = 0) -> None:
    if hasattr(module, "weight") and _get_tensor(module.weight):
        nn_init.uniform_(module.weight, a, b)
    if hasattr(module, "bias") and _get_tensor(module.bias):
        nn_init.constant_(module.bias, bias)


def kaiming_init(
    module: nn.Module,
    a: float = 0,
    mode: str = "fan_out",
    nonlinearity: str = "relu",
    bias: float = 0,
    distribution: str = "normal",
) -> None:
    assert distribution in ["uniform", "normal"], "distribution must be 'uniform' or 'normal'"
    if hasattr(module, "weight") and _get_tensor(module.weight):
        if distribution == "uniform":
            nn_init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn_init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, "bias") and _get_tensor(module.bias):
        nn_init.constant_(module.bias, bias)


def caffe2_xavier_init(module: nn.Module, bias: float = 0) -> None:
    kaiming_init(module, a=1, mode="fan_in", nonlinearity="leaky_relu", bias=bias, distribution="uniform")


def bias_init_with_prob(prior_prob: float) -> float:
    return float(-math.log((1 - prior_prob) / prior_prob))
