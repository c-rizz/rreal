import torch as th
import torch.nn as nn
from typing import Tuple, Callable

from rreal.utils.utils import build_mlp_net
from rreal.nets.ConvNet import ConvNet
from rreal.nets.BigConvNet import BigConvNet
import adarl.utils.dbg.ggLog as ggLog
from rreal.nets.ResNet18 import ResNet18
from rreal.nets.MobileNetV3 import MobileNetV3

class Image_encoder(nn.Module):
    def __init__(self,  image_channels_num : int = 1,
                        net_input_width : int = 64,
                        net_input_height : int = 64,
                        backbone : str = "conv",
                        checkDimensions : bool = True,
                        torchDevice : str | th.device = "cuda",
                        use_coord_conv : bool = False,
                        fc_net_arch = "identity",
                        output_size = None,
                        last_activation_class : Callable[[], nn.Module] = th.nn.Identity,
                        layerNorm : bool = False,
                        use_batchnorm = True,
                        use_weightnorm = True):
        super().__init__()
        self._checkDimensions = checkDimensions
        self._input_width  = net_input_width
        self._input_height = net_input_height
        self._input_channels = image_channels_num
        self._fc_net_arch = fc_net_arch
        self._layerNorm = layerNorm
        self._backbone = backbone.lower()

        # print(f"Building encoder for size {net_input_width}x{net_input_height}x{image_channels_num}")

        if self._input_height!=self._input_width:
            raise NotImplementedError(f"Only square images are supported, got {self._input_width}x{self._input_height}")
        enc_layers_kernel_sizes = None
        enc_layers_paddings = None
        if self._backbone=="mobilenetv3":
            encoder_head = MobileNetV3( image_channels = self._input_channels,
                                                        image_width = self._input_width,
                                                        image_height = self._input_height)
        elif self._backbone=="resnet18":
            encoder_head = ResNet18(image_channels = self._input_channels,
                                                    image_width = self._input_width,
                                                    image_height = self._input_height)
        elif self._backbone=="conv":
            if self._input_height==64 or self._input_width==64:
                enc_layers_channels_num = [32, 64, 128, 256]
            elif self._input_height==84 or self._input_width==84:
                enc_layers_channels_num = [32, 64, 128, 192, 256]
                enc_layers_strides      = [2,   1,  2,  2,  2]
            elif self._input_height==128 or self._input_width==128:
                enc_layers_channels_num = [32, 64, 128, 192, 256]
            else:
                raise NotImplementedError(f"Currently only 64x64 or 128x128 network input is supported with convnet backbone. You asked for height={self._input_height}, width={self._input_width}")
            encoder_head = ConvNet( image_channels = self._input_channels,
                                    image_width = self._input_width,
                                    image_height = self._input_height,
                                    filters_number = enc_layers_channels_num,
                                    torchDevice = torchDevice,
                                    use_coord_conv = use_coord_conv)
        elif self._backbone=="conv_small":
            enc_layers_strides = None
            if self._input_height==64 or self._input_width==64:
                enc_layers_channels_num = [32, 32, 32, 32]
                enc_layers_strides      = [2,   2,  1,  1]
            elif self._input_height==84 or self._input_width==84:
                enc_layers_channels_num = [32, 32, 32, 32, 32]
                # enc_layers_strides      = [2,   2,  2,  2,  2]
                enc_layers_strides      = [2,   2,  1,  1,  1]
            elif self._input_height==128 or self._input_width==128:
                enc_layers_channels_num = [32, 32, 32, 32, 32]
                enc_layers_strides      = [2,   2,  2,  1,  1]
            else:
                raise NotImplementedError(f"Resolution not supported by {self._backbone}. You asked for height={self._input_height}, width={self._input_width}")
            encoder_head = ConvNet( image_channels = self._input_channels,
                                    image_width = self._input_width,
                                    image_height = self._input_height,
                                    filters_number = enc_layers_channels_num,
                                    strides = enc_layers_strides,
                                    use_coord_conv = use_coord_conv,
                                    use_batchnorm = use_batchnorm,
                                    torchDevice = torchDevice)        
        elif self._backbone=="conv_small2":
            enc_layers_strides = None
            if self._input_height==84 or self._input_width==84:
                enc_layers_channels_num = [32, 32, 64, 64]
                # enc_layers_strides      = [2,   2,  2,  2,  2]
                enc_layers_strides      = [2,   2,  2,  1]
                # 
            elif self._input_height==64 or self._input_width==64:
                enc_layers_channels_num = [32, 32, 64, 64]
                # enc_layers_strides      = [2,   2,  2,  2,  2]
                enc_layers_strides      = [2,   2,  2,  1]
            else:
                raise NotImplementedError(f"Resolution not supported by {self._backbone}. You asked for height={self._input_height}, width={self._input_width}")
            encoder_head = ConvNet( image_channels = self._input_channels,
                                    image_width = self._input_width,
                                    image_height = self._input_height,
                                    filters_number = enc_layers_channels_num,
                                    strides = enc_layers_strides,
                                    use_coord_conv = use_coord_conv,
                                    use_batchnorm = use_batchnorm,
                                    torchDevice = torchDevice)
        elif self._backbone=="conv_extrasmall":
            enc_layers_strides = None
            if self._input_height==84 or self._input_width==84:
                enc_layers_channels_num = [8, 16, 16, 16]
                enc_layers_strides      = [2,  2,  2,  2]
                enc_layers_paddings     = [0,       1,           1,           1]
                #                          84x84 -> 41x41 + 2 -> 21x21 + 2 -> 11x11 -> 6x6
            elif self._input_height==64 or self._input_width==64:
                enc_layers_channels_num = [8, 16, 16, 32]
                enc_layers_strides      = [2,  2,  2,  1]
            else:
                raise NotImplementedError(f"Resolution not supported by {self._backbone}. You asked for height={self._input_height}, width={self._input_width}")
            encoder_head = ConvNet( image_channels = self._input_channels,
                                    image_width = self._input_width,
                                    image_height = self._input_height,
                                    filters_number = enc_layers_channels_num,
                                    paddings = enc_layers_paddings,
                                    kernel_sizes = enc_layers_kernel_sizes,
                                    strides = enc_layers_strides,
                                    use_coord_conv = use_coord_conv,
                                    use_batchnorm = use_batchnorm,
                                    torchDevice = torchDevice)
        elif self._backbone=="bigconv":
            encoder_head =BigConvNet( image_channels = self._input_channels,
                                                    image_width = self._input_width,
                                                    image_height = self._input_height)
        else:
            raise AttributeError(f"Unknown backbone '{self._backbone}'")

        conv_out_shape = encoder_head.output_shape
        self._conv_out_size = 1
        for s in conv_out_shape:
            self._conv_out_size *= s
        if output_size is None:
            output_size = self._conv_out_size
        self._output_size = output_size

        # ggLog.info(f"encoder_head output shape = {conv_out_shape}")

        if self._layerNorm:
            fc_last_act = lambda : th.nn.Sequential(th.nn.LayerNorm(self._output_size),last_activation_class())
        else:
            fc_last_act = last_activation_class

        fc = build_mlp_net(self._fc_net_arch, self._conv_out_size, self._output_size, ensemble_size=1,
                                last_activation_class=fc_last_act, return_ensemble_mean=True,
                                use_weightnorm=use_weightnorm)

        self._net = th.nn.Sequential(encoder_head, th.nn.Flatten(), fc)

    @property
    def output_shape(self):
        return (self._conv_out_size,)

    def input_width(self):
        return self._input_width

    def input_height(self):
        return self._input_height

    def input_channels(self):
        return self._input_channels

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """

        Parameters
        ----------
        x : th.Tensor
            Batch of input images. Shape should be [batch_size,input_channels,input_height,input_width]
 
        Returns
        -------
        List[th.Tensor]
            Two batches, containing latent space mean and log-variance for each input image.
            Size of each of the two batches will be (batch_size, latent_space_size)
        """
        if self._checkDimensions:
            assert x.size() == (x.size()[0],self._input_channels,self._input_height, self._input_width), f"Image batch should have size {(x.size()[0],self._input_channels,self._input_height, self._input_width)}, but it is {x.size()}"

        # ggLog.info(f"Image_encoder.forward(): x.size() = {x.size()}")
        enc_out = self._net(x)
        if self._checkDimensions:
            batch_size = x.size()[0]
            assert len(enc_out.size())==2, "Encoder output should have two dimensions (because it's a batch)"
            assert enc_out.size()==(batch_size, self._output_size), f"Encoder outputs should have size (batch_size, output_size)=={(batch_size,self._output_size)}, but enc_out.size() = {enc_out.size()}"
            
        return enc_out
            

    