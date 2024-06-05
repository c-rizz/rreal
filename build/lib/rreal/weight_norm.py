from enum import Enum, auto

import torch
from torch import Tensor
from torch.nn.utils import parametrize
from torch.nn.modules import Module
from torch.nn import functional as F

from typing import Optional


# Taking new weightnorm from pytorch main. Once it is officially available this file should be removed



class _WeightNorm(Module):
    def __init__(
        self,
        dim: Optional[int] = 0,
    ) -> None:
        super().__init__()
        if dim is None:
            dim = -1
        self.dim = dim

    def forward(self, weight_g, weight_v):
        return torch._weight_norm(weight_v, weight_g, self.dim)

    def right_inverse(self, weight):
        weight_g = torch.norm_except_dim(weight, 2, self.dim)
        weight_v = weight

        return weight_g, weight_v


def weight_norm(module: Module, name: str = 'weight', dim: int = 0):
    r"""Applies weight normalization to a parameter in the given module.

    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}

    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by :attr:`name` with two parameters: one specifying the magnitude
    and one specifying the direction.

    By default, with ``dim=0``, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    ``dim=None``.

    See https://arxiv.org/abs/1602.07868

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm

    Returns:
        The original module with the weight norm hook

    Example::

        >>> m = weight_norm(nn.Linear(20, 40), name='weight')
        >>> m
        ParametrizedLinear(
          in_features=20, out_features=40, bias=True
          (parametrizations): ModuleDict(
            (weight): ParametrizationList(
              (0): _WeightNorm()
            )
          )
        )
        >>> m.parametrizations.weight.original0.size()
        torch.Size([40, 1])
        >>> m.parametrizations.weight.original1.size()
        torch.Size([40, 20])

    """
    _weight_norm = _WeightNorm(dim)
    parametrize.register_parametrization(module, name, _weight_norm, unsafe=True)

    def _weight_norm_compat_hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        g_key = f"{prefix}{name}_g"
        v_key = f"{prefix}{name}_v"
        if g_key in state_dict and v_key in state_dict:
            original0 = state_dict.pop(g_key)
            original1 = state_dict.pop(v_key)
            state_dict[f"{prefix}parametrizations.{name}.original0"] = original0
            state_dict[f"{prefix}parametrizations.{name}.original1"] = original1
    module._register_load_state_dict_pre_hook(_weight_norm_compat_hook)
    return module