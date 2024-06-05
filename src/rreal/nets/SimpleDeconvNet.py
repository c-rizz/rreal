import torch as th
import torch.nn as nn


class SimpleDeconvNet(nn.Module):

    def __init__(self):
        super().__init__()
        self._input_width  = 8
        self._input_height = 8
        self._input_channels = 16

        self._output_width  = 64
        self._output_height = 64
        self._output_channels = 3

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
        b6_channels = self._input_channels
        b6_kernelSize = 5
        b6_stride = 1

        self._in_size = (self._input_channels, self._input_height, self._input_width)
        self._out_size = (self._output_channels, self._output_height, self._output_width)

        self.decoder = nn.Sequential(   #Block6
                                        nn.ConvTranspose2d(in_channels = b6_channels, out_channels = b4_channels, kernel_size = b6_kernelSize, stride = b6_stride),
                                        nn.BatchNorm2d(b4_channels),
                                        nn.ReLU(),
                                        #Block5
                                        nn.Upsample(scale_factor=b5_maxpool_size,mode="nearest"),
                                        #Block4
                                        nn.ConvTranspose2d(in_channels = b4_channels, out_channels = b2_channels, kernel_size = b4_kernelSize, stride = b4_stride),
                                        nn.BatchNorm2d(b2_channels),
                                        nn.ReLU(),
                                        #Block3
                                        nn.Upsample(scale_factor=b3_maxpool_size,mode="nearest"),
                                        #Block2
                                        nn.ConvTranspose2d(in_channels = b2_channels, out_channels = b1_channels, kernel_size = b2_kernelSize, stride = b2_stride),
                                        nn.BatchNorm2d(b1_channels),
                                        nn.ReLU(),
                                        #Block1
                                        nn.ConvTranspose2d(in_channels = b1_channels, out_channels = self._output_channels, kernel_size = b1_kernelSize, stride = b1_stride),
                                        nn.BatchNorm2d(self._output_channels),
                                        nn.Sigmoid()
                                        )


    @property
    def output_shape(self):
        return self._out_size

    @property
    def input_size(self):
        return self._in_size

    def forward(self, x : th.Tensor) -> th.Tensor:
        return self.decoder(x)