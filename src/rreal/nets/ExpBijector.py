import torch as th
import torch.nn as nn

class ExpBijector(nn.Module):
    """ A bijector that can apply a linear transformation and its inverse.
    The underlying linear transformation is parameterized as a matrix exponential, W = expm(M), where M is a learnable matrix.
    """
    def __init__(self, dim, bias=True):
        super().__init__()
        self.M = nn.Parameter(th.zeros(dim, dim))   # starts at identity
        self.b = nn.Parameter(th.zeros(dim)) if bias else None

    def forward(self, x) -> th.Tensor:
        W = th.linalg.matrix_exp(self.M)
        y = x @ W.T
        return y + self.b if self.b is not None else y

    def inverse(self, y) -> th.Tensor:
        if self.b is not None:
            y = y - self.b
        Winv = th.linalg.matrix_exp(-self.M)        # expm(−M) = W⁻¹
        return y @ Winv.T