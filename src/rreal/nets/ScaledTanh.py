import torch as th
from typing import Final

class ScaledTanh(th.nn.Tanh):
    def __init__(self, scale = None, xscale = None, yscale = None):
        super().__init__()
        if scale is not None:
            if xscale is not None or yscale is not None:
                raise AttributeError(f"You cannot specify both scale and (xscale, yscale)")
            xscale = 1/scale
            yscale = scale
        self.register_buffer('_xscale', th.tensor(xscale))
        self.register_buffer('_yscale', th.tensor(yscale))

    def forward(self, x):
        return self._yscale*super().forward(x*self._xscale)