import torch as th
import torch.nn as nn

class SimpleConvNet(nn.Module):

    def __init__(self, image_channels_num : int = 1, image_width : int = 64, image_height : int = 64):
        super().__init__()
        self._input_width  = image_width
        self._input_height = image_height
        self._input_channels = image_channels_num

        if self._input_height!=64 or self._input_width!=64:
            raise NotImplementedError("Currently only 64x64 network input is supported")

        #HxW
        b1_channels = 32
        b1_kernelSize = 5
        b1_stride = 1

        #(H-4)x(W-4)
        b2_channels = 32
        b2_kernelSize = 5
        b2_stride = 1

        #(H-8)x(W-8)
        b3_maxpool_size = 2

        #(H-8)/2x(W-8)/2
        b4_channels = 16
        b4_kernelSize = 5
        b4_stride = 1

        #(H-8)x(W-8)
        b5_maxpool_size = 2

        #(H-8)/2x(W-8)/2
        b6_channels = 16
        b6_kernelSize = 5
        b6_stride = 1


        #((H-8)/2-4)x((W-8)/2-4)
        self._out_size = (b6_channels,8,8)        

        self._encoder = nn.Sequential(   #Block1
                                        nn.Conv2d(in_channels = self._input_channels, out_channels = b1_channels, kernel_size=b1_kernelSize, stride=b1_stride),
                                        nn.BatchNorm2d(b1_channels),
                                        nn.ReLU(),
                                        #Block2
                                        nn.Conv2d(in_channels = b1_channels, out_channels = b2_channels, kernel_size=b2_kernelSize, stride=b2_stride),
                                        nn.BatchNorm2d(b2_channels),
                                        nn.ReLU(),
                                        #Block3
                                        nn.MaxPool2d(b3_maxpool_size),
                                        #Block4
                                        nn.Conv2d(in_channels = b2_channels, out_channels = b4_channels, kernel_size=b4_kernelSize, stride=b4_stride),
                                        nn.BatchNorm2d(b4_channels),
                                        nn.ReLU(),
                                        #Block5
                                        nn.MaxPool2d(b5_maxpool_size),
                                        #Block6
                                        nn.Conv2d(in_channels = b4_channels, out_channels = b6_channels, kernel_size=b6_kernelSize, stride=b6_stride),
                                        nn.BatchNorm2d(b6_channels),
                                        nn.ReLU())

    @property
    def output_shape(self):
        return self._out_size

    def forward(self, x : th.Tensor) -> th.Tensor:
        return self._encoder(x)