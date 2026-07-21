import torch as th
import torch.nn as nn
import torchvision
from torchvision.transforms.functional import InterpolationMode

from rreal.nets.DeconvNet import DeconvNet
from rreal.nets.Parallel import Parallel
from rreal.utils.utils import build_mlp_net


class Image_decoder(nn.Module):
    def __init__(self,  latent_space_size : int,
                        output_channels_num : int = 1,
                        net_output_width : int = 64,
                        net_output_height : int = 64,
                        checkDimensions : bool = True,
                        torchDevice : str | th.device = "cuda",
                        backbone = "conv",
                        ensemble_size = 1,
                        use_batchnorm = True,
                        use_weightnorm = False,
                        pixel_range_scaling = 1.1,
                        learn_background = True,
                        fc_head_layer_sizes = []):
        super().__init__()
        # ggLog.info(f"recieved learn_background = {learn_background}")
        self._checkDimensions = checkDimensions
        self._output_width  = net_output_width
        self._output_height = net_output_height
        self._output_channels = output_channels_num
        self._latent_space_size = latent_space_size
        self._fc_head_layer_sizes = fc_head_layer_sizes
        self._resizeToOutput = None
        self._backbone = backbone
        self._ensemble_size = ensemble_size
        self._output_shape_chw = (self._output_channels, self._output_height, self._output_width)
        self._pixel_range_scaling = pixel_range_scaling
        self._learn_background = learn_background


        if self._output_height == self._output_width == 0:
            return
        
        if self._output_height!=self._output_width:
            raise NotImplementedError("Only square images are supported")

        dec_layers_scales = None
        dec_layers_strides = None
        dec_kernel_sizes = None
        dec_paddings = None
        if self._backbone=="conv":
            if self._output_height==64 and self._output_width==64:
                dec_layers_channels_num = [256, 128, 64, 32]
                dec_kernel_sizes =        [  3,   3,  3,  5]
                self._net_output_size = self._output_width
                dec_input_shape = (dec_layers_channels_num[0],4,4)
            elif self._output_height==84 and self._output_width==84:
                # dec_layers_channels_num = [256, 128, 64, 32, 32]
                # dec_layers_scales  = [2,2,2,2,2]
                # self._net_output_size = 160
                # dec_input_shape = (dec_layers_channels_num[0],5,5)
                # intMode = InterpolationMode.NEAREST # if th.is_deterministic() else InterpolationMode.BILINEAR  # Bilinear does not support deterministic
                # self._resizeToOutput = torchvision.transforms.Resize((self._output_height,self._output_width),
                #                                                         interpolation=intMode)
                dec_layers_channels_num = [256, 128, 64,  32]
                dec_layers_scales =       [  2,   2,  2, 2.1]
                dec_kernel_sizes =        [  3,   3,  3,   3]
                dec_paddings =            [  1,   1,  1,   1]
                self._net_output_size = self._output_width
                dec_input_shape = (dec_layers_channels_num[0],5,5)
            elif self._output_height==128 and self._output_width==128:
                dec_layers_channels_num = [256, 192, 128, 64, 32]
                dec_kernel_sizes =        [  3,   3,  3,   3,  3]
                self._net_output_size = self._output_width
                dec_input_shape = (dec_layers_channels_num[0],4,4)
            elif self._output_height==256 and self._output_width==256:
                dec_layers_channels_num = [256, 192, 128, 64, 64, 32]
                dec_kernel_sizes =        [  3,   3,   3,  3,  3,  3]
                self._net_output_size = self._output_width
                dec_input_shape = (dec_layers_channels_num[0],4,4)
            elif self._output_height==224 and self._output_width==224:
                dec_layers_channels_num = [256, 192, 128, 64, 64, 32]
                dec_kernel_sizes =        [  3,   3,   3,  3,  3,  3]
                self._net_output_size = 256
                dec_input_shape = (dec_layers_channels_num[0],4,4)
                intMode = InterpolationMode.NEAREST # if th.is_deterministic() else InterpolationMode.BILINEAR  # Bilinear does not support deterministic
                self._resizeToOutput = torchvision.transforms.Resize((self._output_height,self._output_width),
                                                                        interpolation=intMode)
            else:
                raise NotImplementedError(f"Requested network output size is not supported. You asked for backbone='{backbone}' height={self._output_height}, width={self._output_width}")
        elif self._backbone=="conv_small":
            if self._output_height==64 and self._output_width==64:
                dec_layers_channels_num = [32, 32, 32, 32]
                dec_kernel_sizes =        [ 3,  3,  3,  3]
                self._net_output_size = self._output_width
                dec_input_shape = (dec_layers_channels_num[0],4,4)
            elif self._output_height==84 and self._output_width==84:
                # dec_layers_channels_num = [256, 128, 64, 32, 32]
                # dec_layers_scales  = [2,2,2,2,2]
                # self._net_output_size = 160
                # dec_input_shape = (dec_layers_channels_num[0],5,5)
                # intMode = InterpolationMode.NEAREST # if th.is_deterministic() else InterpolationMode.BILINEAR  # Bilinear does not support deterministic
                # self._resizeToOutput = torchvision.transforms.Resize((self._output_height,self._output_width),
                #                                                         interpolation=intMode)
                dec_layers_channels_num = [32, 32, 32,  32]
                dec_layers_scales =       [ 2,  2,  2, 2.1]
                dec_kernel_sizes =        [ 3,  3,  3,   3]
                dec_paddings =            [ 1,  1,  1,   1]
                self._net_output_size = self._output_width
                dec_input_shape = (dec_layers_channels_num[0],5,5)
            elif self._output_height==128 and self._output_width==128:
                dec_layers_channels_num = [32, 32, 32, 32, 32]
                dec_kernel_sizes =        [ 3,  3,  3,  3,  3]
                self._net_output_size = self._output_width
                dec_input_shape = (dec_layers_channels_num[0],4,4)
            elif self._output_height==256 and self._output_width==256:
                dec_layers_channels_num = [32, 32, 32, 32, 32, 32]
                dec_kernel_sizes =        [ 3,  3,  3,  3,  3,  3]
                self._net_output_size = self._output_width
                dec_input_shape = (dec_layers_channels_num[0],4,4)
            elif self._output_height==224 and self._output_width==224:
                dec_layers_channels_num = [32, 32, 32, 32, 32, 32]
                dec_kernel_sizes =        [ 3,  3,  3,  3,  3,  3]
                self._net_output_size = 256
                dec_input_shape = (dec_layers_channels_num[0],4,4)
                intMode = InterpolationMode.NEAREST # if th.is_deterministic() else InterpolationMode.BILINEAR  # Bilinear does not support deterministic
                self._resizeToOutput = torchvision.transforms.Resize((self._output_height,self._output_width),
                                                                        interpolation=intMode)
            else:
                raise NotImplementedError(f"Requested network output size is not supported. You asked for backbone='{backbone}' height={self._output_height}, width={self._output_width}")
        elif self._backbone=="conv_extrasmall":
            if self._output_height==64 and self._output_width==64:
                dec_layers_channels_num = [16, 16, 16, 8]
                dec_kernel_sizes =        [ 3,  3,  3,  3]
                self._net_output_size = self._output_width
                dec_input_shape = (dec_layers_channels_num[0],4,4)
            elif self._output_height==84 and self._output_width==84:
                dec_layers_channels_num = [16, 16, 16,  8]
                dec_layers_scales =       [ 2,  2,  2, 2.1]
                dec_kernel_sizes =        [ 3,  3,  3,   3]
                dec_paddings =            [ 1,  1,  1,   1]
                self._net_output_size = self._output_width
                dec_input_shape = (dec_layers_channels_num[0],5,5)
            else:
                raise NotImplementedError(f"Requested network output size is not supported. You asked for backbone='{backbone}' height={self._output_height}, width={self._output_width}")
        elif self._backbone=="conv_smaller":
            if self._output_height==64 and self._output_width==64:
                dec_layers_channels_num = [32, 32, 32, 32]
                dec_kernel_sizes =        [ 3,  3,  3,  3]
                self._net_output_size = self._output_width
                dec_input_shape = (dec_layers_channels_num[0],4,4)
            elif self._output_height==84 and self._output_width==84:
                dec_layers_channels_num = [32, 32, 32, 32]
                dec_layers_scales =       [ 2,  2,  2, 2.1]
                dec_kernel_sizes =        [ 3,  3,  3,   3]
                dec_paddings =            [ 1,  1,  1,   1]
                self._net_output_size = self._output_width
                dec_input_shape = (dec_layers_channels_num[0],5,5)
            else:
                raise NotImplementedError(f"Requested network output size is not supported. You asked for backbone='{backbone}' height={self._output_height}, width={self._output_width}")
        else:
            raise RuntimeError(f"Unknown backbone {self._backbone}")

        if self._learn_background:
            self._deconv_channels = self._output_channels+1 # add an alpha channel to the deconv output
            self._background = th.nn.Parameter(th.zeros(size=self._output_shape_chw, dtype = th.float32))
        else:
            self._deconv_channels = self._output_channels
        # ggLog.info(f"dec_input_shape = {dec_input_shape}")
        # ggLog.info(f"dec_layers_channels_num = {dec_layers_channels_num}")
        linear_out_size = dec_input_shape[0]*dec_input_shape[1]*dec_input_shape[2]
        self._fc_head =  build_mlp_net( self._fc_head_layer_sizes,
                                        input_size=self._latent_space_size,
                                        output_size=linear_out_size,
                                        use_weightnorm=use_weightnorm,
                                        last_activation_class=nn.LeakyReLU,
                                        return_ensemble_mean=True)
        
        deconvNet = Parallel([DeconvNet(output_channels = self._deconv_channels,
                                        output_width = self._net_output_size,
                                        output_height = self._net_output_size,
                                        filters_number = dec_layers_channels_num,
                                        scales = dec_layers_scales,
                                        strides = dec_layers_strides,
                                        kernel_sizes = dec_kernel_sizes,
                                        paddings = dec_paddings,
                                        torchDevice = torchDevice,
                                        input_shape_chw = dec_input_shape,
                                        use_batchnorm=use_batchnorm) 
                                for _ in range(self._ensemble_size)],
                                return_mean=True)

        self._decoder  = nn.Sequential(  nn.Unflatten(dim = 1, unflattened_size = dec_input_shape),
                                        deconvNet)
        # ggLog.info(f"{self}")



    @property
    def latent_space_size(self):
        return self._latent_space_size

    @property
    def output_width(self):
        return self._output_width

    @property
    def output_height(self):
        return self._output_height

    @property
    def output_channels(self):
        return self._output_channels

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Decode a batch of images.

        Parameters
        ----------
        x : torch.Tensor
            Batch of encoded images. Size must be (batch_size, self.latent_space_size)

        Returns
        -------
        x : torch.Tensor
            A batch of images. Size will be (batch_size, output_height, output_width)

        """
        if self._checkDimensions:
            assert x.size() == (x.size()[0],self._latent_space_size), f"latent-vecotrs batch should have size {(x.size()[0],self._latent_space_size)}, not {x.size()}"

        batch_size = x.size()[0]
        if self._output_height == self._output_width == 0:
            return th.empty(size=(batch_size,)+self._output_shape_chw).to(x.device)

        h = self._fc_head(x)
        img = self._decoder(h)
        if self._learn_background:
            # should use transparency and blending
            # img = th.nn.functional.tanh(2*img + th.nn.functional.tanh(self._background))
            # alpha = img[:,-1,:,:].unsqueeze(1)
            alpha = (img[:,-1,:,:].unsqueeze(1) + 1)/2
            squashed_bg = th.nn.functional.tanh(self._background).unsqueeze(0)
            # ggLog.info(f"Size input = {img.size()}")
            # ggLog.info(f"Size alpha = {alpha.size()}")
            # ggLog.info(f"Size img[:,:-1,:,:] = {img[:,:-1,:,:].size()}")
            # ggLog.info(f"Size squashed_bg = {squashed_bg.size()}")
            premulimg = img[:,:-1,:,:] * alpha
            # ggLog.info(f"Size premulimg = {img.size()}")
            premulbg = squashed_bg*(1-alpha)
            # ggLog.info(f"Size prremulbg = {premulbg.size()}")
            img = premulimg  + premulbg
            # ggLog.info(f"Size img = {img.size()}")
        result = img*self._pixel_range_scaling  # Scaling allows tanh to reach +1 and -1

        # ggLog.info(f"result raw size = {result.size()}")
        if self._resizeToOutput is not None:
            result = self._resizeToOutput(result)
        # ggLog.info(f"result resized size = {result.size()}")
        
        return result
            


    
