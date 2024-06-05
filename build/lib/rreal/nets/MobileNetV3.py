from configparser import Interpolation
import torch as th
import torch.nn as nn
from typing import Tuple, List
import torchvision
import lr_gym.utils.dbg.dbg_img as dbg_img
from autoencoding_rl.utils import tensorToHumanCvImageRgb
import lr_gym.utils.dbg.ggLog as ggLog
from torchvision.transforms.functional import InterpolationMode

class MobileNetV3(nn.Module):
    def __init__(self,  image_channels : int = 1,
                        image_width : int = 64,
                        image_height : int = 64,
                        train_conv_part : bool = True):
        super().__init__()
        self._image_width  = image_width
        self._image_height = image_height
        self._image_channels = image_channels

        if self._image_height!=self._image_width:
            raise NotImplementedError("Only square images are supported")
            
        
        self._inputResize = torchvision.transforms.Resize((224,224), interpolation=InterpolationMode.NEAREST)

        self._mobilenet = torchvision.models.mobilenet_v3_small(pretrained=True)
        if not train_conv_part:
            for param in self._mobilenet.parameters():
                param.requires_grad = False
        self._feat_out_size = self._mobilenet.classifier[0].in_features
        # self._model_conv.fc = nn.Linear(self._model_conv.fc.in_features, self._out_size[0])
        self._mobilenet.classifier = nn.Flatten(start_dim=1)

        self._out_size = (self._feat_out_size,)
    
    @property
    def output_shape(self):
        return self._out_size

    def forward(self, x : th.Tensor) -> th.Tensor:
        x = (x+1)/2
        # ggLog.info(f"x.size()={x.size()}")
        upscaled_img = self._inputResize(x)
        # ggLog.info(f"upscaled_img.size()={upscaled_img.size()}")
        # dbg_img.helper.publishDbgImg("mobilenet_input", tensorToHumanCvImageRgb(upscaled_img[0]))
        return self._mobilenet(upscaled_img)
