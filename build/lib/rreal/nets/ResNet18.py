import torch as th
import torch.nn as nn
from typing import Tuple, List
import torchvision

class ResNet18(nn.Module):
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
            

        self._inputResize = torchvision.transforms.Resize((224,224))

        self._model_conv = torchvision.models.resnet18(pretrained=True)
        if not train_conv_part:
            for param in self._model_conv.parameters():
                param.requires_grad = False
        self._conv_out_size = self._model_conv.fc.in_features
        # self._model_conv.fc = nn.Linear(self._model_conv.fc.in_features, self._out_size[0])
        self._model_conv.fc = nn.Flatten(start_dim=1)

        self._out_size = (self._conv_out_size,)
    
    @property
    def output_shape(self):
        return self._out_size

    def forward(self, x : th.Tensor) -> th.Tensor:
        x = (x+1)/2
        return self._model_conv(self._inputResize(x))
