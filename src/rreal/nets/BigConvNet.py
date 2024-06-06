import torch as th
import torch.nn as nn
from typing import Tuple, List
import torchvision
import adarl.utils.dbg.dbg_img as dbg_img
from autoencoding_rl.utils import tensorToHumanCvImageRgb
from rreal.nets.ConvNet import ConvNet

class BigConvNet(nn.Module):
    def __init__(self,  image_channels : int = 1,
                        image_width : int = 64,
                        image_height : int = 64,
                        train_conv_part : bool = False):
        super().__init__()
        self._image_width  = image_width
        self._image_height = image_height
        self._image_channels = image_channels

        if self._image_height!=self._image_width:
            raise NotImplementedError("Only square images are supported")
            

        self._inputResize = torchvision.transforms.Resize((256,256))

        self._convnet = ConvNet( image_channels = self._image_channels,
                                 image_width = 256,
                                 image_height = 256,
                                 filters_number = [32, 64, 128, 192, 256,512])
        if not train_conv_part:
            for param in self._convnet.parameters():
                param.requires_grad = False

        self._out_size = self._convnet.output_shape
    
    @property
    def output_shape(self):
        return self._out_size

    def forward(self, x : th.Tensor) -> th.Tensor:
        x = (x+1)/2
        upscaled_img = self._inputResize(x)
        # dbg_img.helper.publishDbgImg("net_input", tensorToHumanCvImageRgb(upscaled_img[0]))
        return self._convnet(upscaled_img)
