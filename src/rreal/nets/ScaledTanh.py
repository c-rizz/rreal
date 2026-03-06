import torch as th
from typing import Final

class ScaledTanh(th.nn.Tanh):
    def __init__(self, scale = None, xscale = None, yscale = None):
        """ A scaled version of the Tanh activation, allowing to squash at different ranges.
        The actual operation performed is yscale*tanh(xscale*x). You can specify the scale with 
        either the scale argument (which will set xscale to 1/scale and yscale to scale) or
        by specifying xscale and yscale directly.

        Parameters
        ----------
        scale : _type_, optional
            Sets xscale to 1/scale and yscale to scale, by default None
        xscale : _type_, optional
            Sets the scale for the input, by default None
        yscale : _type_, optional
            Sets the scale for the output, by default None

        Raises
        ------
        AttributeError
            _description_
        """
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