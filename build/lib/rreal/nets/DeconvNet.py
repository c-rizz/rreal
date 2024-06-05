from matplotlib import scale
import torch as th
import torch.nn as nn
from typing import List
import numpy as np
import lr_gym.utils.dbg.ggLog as ggLog

# from torchsummary import summary
class ResizeConvolution(nn.Module):
    # See: Deconvolution and Checkerboard Artifacts, Augustus Odena et al., 2016,  https://distill.pub/2016/deconv-checkerboard/
    def __init__(self,  in_channels : int,
                        out_channels : int, 
                        scale : int = 2,
                        kernel_size: int = 3,
                        stride : int = 1,
                        padding:int = 1,
                        bias=True):
        super().__init__()
        self._net = nn.Sequential(  nn.Upsample(scale_factor = scale, mode='nearest', recompute_scale_factor=True),
                                    nn.Conv2d(  in_channels, out_channels,
                                                kernel_size=kernel_size, stride=stride,
                                                padding=padding, padding_mode="zeros",
                                                bias = bias)) #Other padding don't support determinism
    def forward(self, x: th.Tensor):
        return self._net(x)


class DeconvNet(nn.Module):
    def __init__(self,  output_channels : int = 1,
                        output_width : int = 64,
                        output_height : int = 64,
                        input_shape_chw : int = (256,4,4),
                        filters_number : List[int] = [256,128,64,32],
                        strides : List[int] = None,
                        kernel_sizes : List[int] = None,
                        paddings : List[int] = None,
                        scales : List[int] = None,
                        torchDevice : str = "cuda",
                        use_batchnorm = True):
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
        #Build the cnn blocks from last to first
        out_channels = self._output_channels
        filters_number = filters_number[::-1] # Reverse, but not inplace
        strides = strides[::-1]
        kernel_sizes = kernel_sizes[::-1]
        paddings = paddings[::-1]
        scales = scales[::-1]

        for i in range(len(filters_number)):
            ch_num = filters_number[i]
            block : List[nn.Module] = []
            convLayer : nn.Module = ResizeConvolution(ch_num, out_channels,
                                                        scale=scales[i],
                                                        kernel_size=kernel_sizes[i],
                                                        stride=strides[i],
                                                        padding=paddings[i],
                                                        bias = i==0) #Only enable bias on last layer (others have batchnorm)
            block.append(convLayer)
            if i > 0: #Normal blocks have batch normalization and leakyRelu
                if use_batchnorm:
                    block.append(nn.BatchNorm2d(out_channels))
                block.append(nn.LeakyReLU())
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