import torch as th
import torch.nn as nn
from typing import Tuple, List, Callable, Literal
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
                        torchDevice : str | th.device = "cuda",
                        use_coord_conv = True,
                        use_batchnorm = True,
                        activation_class : Callable[[],nn.Module] = nn.LeakyReLU,
                        norm_class : Callable[[int],nn.Module] | None = None,
                        downsample : Literal["stride","maxpool"] = "stride"):
        """
        Parameters
        ----------
        activation_class : Callable[[],nn.Module]
            Activation placed after the normalization of each block.
        norm_class : Callable[[int],nn.Module] | None
            Normalization of each block, called with the block's channel count. When None the
            block falls back to BatchNorm2d if use_batchnorm is set, and to no normalization
            otherwise, which is the historical behaviour.
        downsample : Literal["stride","maxpool"]
            How each block reduces the spatial size. "stride" gives the convolution the block's
            stride, which is the historical behaviour. "maxpool" instead keeps the convolution at
            stride 1 and max pools by the block's stride right after it, which is what DreamerV3
            does when its `strided` option is off (its default). Both give the same spatial sizes.
        """
        super().__init__()
        self._image_width  = image_width
        self._image_height = image_height
        self._image_channels = image_channels
        self._torchDevice = torchDevice
        padding_mode = "reflect" # It was "replicate" in the past, but it does not work nice with th.compile (tried patching it but no easy success)

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
                                                        torchDevice=self._torchDevice)
            else:
                convClass = nn.Conv2d
            ch_num = filters_number[i]
            block = []
            block.append(convClass(in_channels=in_channels, out_channels=ch_num,
                              kernel_size=kernel_sizes[i],
                              stride=1 if downsample=="maxpool" else strides[i],
                              padding=paddings[i], bias=False,
                              padding_mode=padding_mode)) #Halves width and height
            if downsample=="maxpool" and strides[i] > 1:
                # Pooling goes between the convolution and the normalization, as in DreamerV3.
                block.append(nn.MaxPool2d(kernel_size=strides[i]))
            if norm_class is not None:
                block.append(norm_class(ch_num))
            elif use_batchnorm:
                block.append(nn.BatchNorm2d(ch_num))
            block.append(activation_class())
            modules.append(nn.Sequential(*block))
            in_channels = ch_num

        self._encoder = nn.Sequential(*modules).to(self._torchDevice)

        # summary(self._encoder, (9,84,84))
        with th.no_grad():
            testImg = th.zeros(size=(1,image_channels,image_height,image_width), device=self._torchDevice)
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
