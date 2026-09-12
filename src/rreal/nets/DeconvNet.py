from matplotlib import scale
import torch as th
import torch.nn as nn
from typing import List
import numpy as np
import adarl.utils.dbg.ggLog as ggLog
from typing import Sequence, Tuple, Callable
import math

class AppendCoordinates(nn.Module):
    _coords : th.Tensor # Declared for the type checker, actually created by register_buffer

    def __init__(self, height : int, width : int):
        super().__init__()
        # Registered as a buffer, and not as a plain attribute, so that it follows the module
        # through .to(device)/.float()/.half(). It is not persistent because it is fully
        # determined by the layer geometry, so it does not belong in the state_dict.
        self.register_buffer("_coords", self._compute_coords(height, width), persistent=False)

    def _compute_coords(self, h : int, w : int) -> th.Tensor:
        # Both channels are expanded to the full (h,w) grid before concatenating: the x channel
        # varies along the width, the y channel along the height, and they must agree on every
        # other dimension to be concatenable.
        y_coords = th.linspace(-1, 1, steps=h).view(1, 1, h, 1).expand(1, 1, h, w)
        x_coords = th.linspace(-1, 1, steps=w).view(1, 1, 1, w).expand(1, 1, h, w)
        return th.cat([x_coords, y_coords], dim=1).contiguous()

    def forward(self, x : th.Tensor) -> th.Tensor:
        # x is of shape (batch_size, channels, height, width)
        batch_size = x.shape[0]
        # Concatenate the coordinates to the input tensor
        coords = self._coords.expand(batch_size, -1, -1, -1)  # Shape: (batch_size, 2, height, width)
        return th.cat([x, coords], dim=1)  # Shape: (batch_size, channels + 2, height, width)

# from torchsummary import summary
class ResizeConvolution(nn.Module):
    # See: Deconvolution and Checkerboard Artifacts, Augustus Odena et al., 2016,  https://distill.pub/2016/deconv-checkerboard/
    def __init__(self,  in_channels : int,
                        out_channels : int,
                        scale : int = 2,
                        kernel_size: int = 3,
                        stride : int = 1,
                        padding:int = 1,
                        bias=True,
                        upsample_interpolation_mode : str = "nearest",
                        input_height : int | None = None,
                        input_width : int | None = None,
                        append_coordinates : bool = False):
        """Upsample followed by a convolution, optionally with coordinate channels.

        Parameters
        ----------
        input_height, input_width : int | None
            Spatial size of the input of this block, i.e. *before* the upsampling. Only needed
            when append_coordinates is set, as the coordinate grids are precomputed from it.
        append_coordinates : bool
            Append two coordinate channels between the upsampling and the convolution, so that
            the convolution can be position dependent. The grids are built at the upsampled
            resolution, which is where the convolution actually reads them.
        """
        super().__init__()
        layers = []
        layers.append(
            nn.Upsample(scale_factor = scale, mode=upsample_interpolation_mode, recompute_scale_factor=True),
        )
        if append_coordinates:
            if input_height is None or input_width is None:
                raise AttributeError("input_height and input_width are required when append_coordinates is set")
            # The grids sit after the upsampling, so they are built at the upsampled size.
            # Scales can be fractional, and nn.Upsample(recompute_scale_factor=True) floors.
            layers.append(AppendCoordinates(height = math.floor(input_height*scale),
                                            width = math.floor(input_width*scale)))
            in_channels += 2
        layers.append(
            nn.Conv2d(  in_channels, out_channels,
                        kernel_size=kernel_size, stride=stride,
                        padding=padding, padding_mode="zeros",
                        bias = bias)
        )
        self._net = nn.Sequential(*layers) #Other padding don't support determinism
    def forward(self, x: th.Tensor):
        return self._net(x)


class DeconvNet(nn.Module):
    def __init__(self,  output_channels : int = 1,
                        output_width : int = 64,
                        output_height : int = 64,
                        input_shape_chw : Sequence[int] = (256,4,4),
                        filters_number : List[int] = [256,128,64,32],
                        upsample_interpolation_mode : str = "nearest",
                        strides : List[int] | None = None,
                        kernel_sizes : List[int] | None = None,
                        paddings : List[int] | None = None,
                        scales : List[int] | None = None,
                        torchDevice : str | th.device = "cuda",
                        use_batchnorm = True,
                        use_coord_conv = False,
                        activation_class : Callable[[],nn.Module] = nn.LeakyReLU,
                        norm_class : Callable[[int],nn.Module] | None = None):
        """
        Parameters
        ----------
        activation_class : Callable[[],nn.Module]
            Activation placed after the normalization of each block. The last block is not
            affected, as it always ends with the output Tanh.
        norm_class : Callable[[int],nn.Module] | None
            Normalization of each block, called with the block's channel count. When None the
            block falls back to BatchNorm2d if use_batchnorm is set, and to no normalization
            otherwise, which is the historical behaviour.
        """
        super().__init__()
        # ggLog.info(f"filters_number = {filters_number}")
        self._output_width  = output_width
        self._output_height = output_height
        self._output_channels = output_channels
        self._input_channels = filters_number[-1]
        required_out_size =  (1, self._output_channels, self._output_height, self._output_width)

        if input_shape_chw[0]!=filters_number[0]:
            raise AttributeError(f"Input shape channels must match first layer channels, but it's respectively {input_shape_chw} and {filters_number}")
        if self._output_height!=self._output_width:
            raise NotImplementedError("Only square images are supported")

        # if self._output_height != self._input_width*(2**len(filters_number)):
        #     raise NotImplementedError(f"The output image size must be {self._input_width}*2^(len(filters_number)), but requested output size is {output_width}x{output_height} and len(filter_number)={len(filters_number)}")

        modules : List[nn.Module] = []

        if strides is None:
            strides = [1]*len(filters_number)
        if kernel_sizes is None:
            kernel_sizes = [3]*len(filters_number)
        if paddings is None:
            paddings = [1]*len(filters_number)
        if scales is None:
            scales = [2]*len(filters_number)

        # Spatial size at the input of each block, in forward order. Computed here, before the
        # per-block parameters get reversed below, as the coordinate grids of each block have to
        # be built at construction time. Each block upsamples by its scale and then convolves.
        blocks_input_hw : List[Tuple[int,int]] = []
        block_h, block_w = int(input_shape_chw[1]), int(input_shape_chw[2])
        for i in range(len(filters_number)):
            blocks_input_hw.append((block_h, block_w))
            # Scales can be fractional, and nn.Upsample(recompute_scale_factor=True) floors.
            upsampled_h = math.floor(block_h*scales[i])
            upsampled_w = math.floor(block_w*scales[i])
            block_h = (upsampled_h + 2*paddings[i] - kernel_sizes[i])//strides[i] + 1
            block_w = (upsampled_w + 2*paddings[i] - kernel_sizes[i])//strides[i] + 1
        if (block_h, block_w) != (self._output_height, self._output_width):
            raise AttributeError(f"The requested block parameters produce a {block_h}x{block_w} output, "
                                 f"but the requested output size is {self._output_height}x{self._output_width}")

        #Build the cnn blocks from last to first
        out_channels = self._output_channels
        filters_number = filters_number[::-1] # Reverse, but not inplace
        strides = strides[::-1]
        kernel_sizes = kernel_sizes[::-1]
        paddings = paddings[::-1]
        scales = scales[::-1]

        for i in range(len(filters_number)):
            ch_num = filters_number[i]
            # The blocks are built in reverse, so block i here is block len-1-i in forward order.
            block_input_h, block_input_w = blocks_input_hw[len(filters_number)-1-i]
            block : List[nn.Module] = []
            convLayer : nn.Module = ResizeConvolution(ch_num, out_channels,
                                                        scale=scales[i],
                                                        kernel_size=kernel_sizes[i],
                                                        stride=strides[i],
                                                        padding=paddings[i],
                                                        upsample_interpolation_mode=upsample_interpolation_mode,
                                                        bias = i==0, #Only enable bias on last layer (others have batchnorm)
                                                        input_height = block_input_h,
                                                        input_width = block_input_w,
                                                        append_coordinates = use_coord_conv)
            block.append(convLayer)
            if i > 0: #Normal blocks have normalization and an activation
                if norm_class is not None:
                    block.append(norm_class(out_channels))
                elif use_batchnorm:
                    block.append(nn.BatchNorm2d(out_channels))
                block.append(activation_class())
            else: #Last layer has tanh activation
                block.append(nn.Tanh())
            blockMod = nn.Sequential(*block)
            modules.append(blockMod)
            out_channels = ch_num




        #Reverse the cnn blocks order
        modules.reverse()
        # print(modules)
        self._decoder = nn.Sequential(*modules).to(torchDevice)

        # summary(self._decoder, input_shape_chw)
        #Check sizes
        self._in_size = input_shape_chw
        # assert deconv_net_input_width == self._input_width, f"Something is wrong resulting input width is different from {self._input_width}"

        with th.no_grad():
            actual_out_shape = self._decoder(th.zeros(size=(1,)+input_shape_chw, device=torchDevice, dtype=th.float32)).shape
        assert actual_out_shape == required_out_size, f"Something is wrong, resulting image has size {actual_out_shape} instead of {required_out_size}"

    @property
    def input_size(self):
        return self._in_size

    def forward(self, x : th.Tensor) -> th.Tensor:
        return self._decoder(x)
