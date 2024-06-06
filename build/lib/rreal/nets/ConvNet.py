import torch as th
import torch.nn as nn
from typing import Tuple, List
from rreal.nets.CoordConv import CoordConv
# from torchsummary import summary 
import adarl.utils.dbg.ggLog as ggLog

class ConvNet(nn.Module):
    def __init__(self,  image_channels : int = 1,
                        image_width : int = 64,
                        image_height : int = 64,
                        filters_number : List[int] = [32, 64, 128, 256],
                        strides : List[int] = None,
                        kernel_sizes : List[int] = None,
                        paddings : List[int] = None,
                        torchDevice : str = "cuda",
                        use_coord_conv = True,
                        use_batchnorm = True):
        super().__init__()
        self._image_width  = image_width
        self._image_height = image_height
        self._image_channels = image_channels
        self._torchDevice = torchDevice

        # self._output_width = 4

        if self._image_height!=self._image_width:
            raise NotImplementedError("Only square images are supported")

        # if self._image_height != self._output_width*(2**len(filters_number)):
        #     raise NotImplementedError(f"The image size must be {self._output_width}*2^(convolution_layers_number), the number of conv layers is determined by len(filters_number")

            
        modules = []

        if strides is None:
            strides = [2]+[1]*(len(filters_number)-1)
        if kernel_sizes is None:
            kernel_sizes = [3]*len(filters_number)
        if paddings is None:
            paddings = [1]*len(filters_number)
        in_channels = self._image_channels


        # Build Encoder
        for i in range(len(filters_number)):
            if use_coord_conv and i==0:
                convClass = lambda **kwargs : CoordConv(**kwargs,
                                                        image_size_chw=(self._image_channels,self._image_height, self._image_width),
                                                        torchDevice=torchDevice)
            else:
                convClass = nn.Conv2d
            ch_num = filters_number[i]
            block = []
            block.append(convClass(in_channels=in_channels, out_channels=ch_num,
                              kernel_size=kernel_sizes[i],
                              stride=strides[i],
                              padding=paddings[i], bias=False)) #Halves width and height
            if use_batchnorm:
                block.append(nn.BatchNorm2d(ch_num))
            block.append(nn.LeakyReLU())
            modules.append(nn.Sequential(*block))
            in_channels = ch_num

        self._encoder = nn.Sequential(*modules).to(torchDevice)

        # summary(self._encoder, (9,84,84))
        with th.no_grad():
            testImg = th.zeros(size=(1,image_channels,image_height,image_width), device=torchDevice)
            outImg = self._encoder(testImg)
        enc_out_shape = outImg.size()
        conv_net_output_width = enc_out_shape[3]
        conv_net_output_height = enc_out_shape[2]
        conv_net_output_channels = enc_out_shape[1]
        self._conv_out_size = (conv_net_output_channels,conv_net_output_height, conv_net_output_width)
        # assert conv_net_output_width == self._output_width, "Something is wrong, output should have size 4x4"
    
    @property
    def output_shape(self):
        return self._conv_out_size

    def forward(self, x : th.Tensor) -> th.Tensor:
        # print(f"Convnet Got x = {x.size()}")
        return self._encoder(x)
