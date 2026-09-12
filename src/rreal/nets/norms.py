import torch as th
import torch.nn as nn


class ChannelRMSNorm2d(nn.Module):
    """RMS normalization over the channel dimension of an NCHW feature map.

    torch.nn.RMSNorm normalizes over the *trailing* dimensions, which for an NCHW tensor would
    mean normalizing over height and width. DreamerV3 normalizes convolutional features over the
    channels (its arrays are NHWC, so channels are the trailing axis there), so this normalizes
    over dim 1 directly instead of permuting to NHWC and back.

    Unlike LayerNorm/BatchNorm there is no mean subtraction and no bias, which is what makes it
    an RMS norm.
    """

    def __init__(self, num_channels : int, eps : float = 1e-6):
        super().__init__()
        self._eps = eps
        self.weight = nn.Parameter(th.ones(num_channels))

    def forward(self, x : th.Tensor) -> th.Tensor:
        # x is of shape (batch_size, channels, height, width)
        inv_rms = th.rsqrt(th.mean(x*x, dim=1, keepdim=True) + self._eps)
        return x*inv_rms*self.weight.view(1, -1, 1, 1)

    def extra_repr(self) -> str:
        return f"num_channels={self.weight.numel()}, eps={self._eps}"
