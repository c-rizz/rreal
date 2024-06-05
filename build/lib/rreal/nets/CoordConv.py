import torch as th
import torch.nn as nn

class CoordConv(nn.Module):
    def __init__(self,  image_size_chw,
                        torchDevice,
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=1,
                        padding=0,
                        dilation=1,
                        groups=1,
                        bias=True,
                        padding_mode='zeros',
                        device=None,
                        dtype=None):
        super().__init__()
        self._torchDevice=torchDevice
        image_width = image_size_chw[2]
        image_height = image_size_chw[2]
        xi = (th.arange(image_width).to(th.float32).repeat(image_width,1)/image_width)*2-1
        yi = (th.arange(image_height).to(th.float32).repeat(image_height,1).transpose(0,1)/image_height)*2-1
        self._position_channels = th.stack([xi,yi]).to(self._torchDevice)
        self._conv = nn.Conv2d(in_channels+2, out_channels, kernel_size, stride, padding,
                                    dilation, groups, bias, padding_mode, device, dtype)

    def forward(self, x : th.Tensor):
        # print(f"Got x = {x.size()}")
        x = th.cat([self._position_channels.repeat(x.size()[0],1,1,1), x], dim=1)
        # print(f"   Made x = {x.size()}")
        return self._conv(x)
