import torch as th
import torch.nn as nn
from rreal.utils.utils import build_mlp_net
from rreal.nets.Image_decoder import Image_decoder
from typing import Callable, List

class Mixed_decoder(nn.Module):
    def __init__(self,
                 input_size : int,
                 common_fc_layer_arch : list[int] | str,
                 common_fc_output_size : int,
                 common_fc_output_activation_class : type[nn.Module],
                 use_weightnorm : bool,
                 vec_decoder_arch : list[int] | str,
                 vec_decoder_output_activation : Callable[[],th.nn.Module],
                 vec_decoder_ensemble_size : int,
                 output_vec_size : int,
                 output_img_size_chw : tuple[int,int,int],
                 img_dec_backbone : str,
                 img_dec_ensemble_size : int,
                 img_dec_fc_layer_sizes : list[int],
                 img_dec_learn_background : bool,
                 torch_device : th.device):
        super().__init__()
        self._input_size = input_size
        self._common_fc_arch = common_fc_layer_arch
        self._common_fc_output_size = common_fc_output_size
        self._common_fc_output_activation_class = common_fc_output_activation_class
        self._use_weightnorm = use_weightnorm
        self._img_dec_backbone = img_dec_backbone
        self._img_dec_ensemble_size = img_dec_ensemble_size
        self._img_dec_learn_background = img_dec_learn_background
        self._img_dec_fc_layer_sizes = img_dec_fc_layer_sizes
        self._vec_decoder_arch = vec_decoder_arch
        self._vec_decoder_output_activation = vec_decoder_output_activation
        self._vec_decoder_ensemble_size = vec_decoder_ensemble_size
        self._output_vec_size = output_vec_size
        self._output_img_size_chw = output_img_size_chw
        self._output_img_channels = output_img_size_chw[0]
        self._output_img_height = output_img_size_chw[1]
        self._output_img_width = output_img_size_chw[2]
        self._torch_device = torch_device

        self._common_fc = build_mlp_net(self._common_fc_arch,
                                        input_size=self._input_size,
                                        output_size=self._common_fc_output_size,
                                        last_activation_class=self._common_fc_output_activation_class,
                                        return_ensemble_mean=True,
                                        use_weightnorm=self._use_weightnorm).to(self._torch_device)
        
        if self._output_vec_size == 0:
            self._vec_decoder = lambda x: th.empty((x.size(0),0), device=x.device)
        else:
            self._vec_decoder = build_mlp_net(  self._vec_decoder_arch,
                                                input_size=self._common_fc_output_size,
                                                output_size=self._output_vec_size,
                                                last_activation_class=self._vec_decoder_output_activation,
                                                return_ensemble_mean=True,
                                                ensemble_size=self._vec_decoder_ensemble_size,
                                                use_weightnorm=self._use_weightnorm,
                                                weight_init_multiplier=0.01).to(self._torch_device)
        
        self._img_decoder = Image_decoder(
                                latent_space_size = self._common_fc_output_size,
                                output_channels_num = self._output_img_channels,
                                net_output_width = self._output_img_width,
                                net_output_height = self._output_img_height,
                                torchDevice = self._torch_device,
                                backbone = self._img_dec_backbone,
                                ensemble_size=self._img_dec_ensemble_size,
                                learn_background=self._img_dec_learn_background,
                                fc_head_layer_sizes=self._img_dec_fc_layer_sizes,)

    def forward(self, x: th.Tensor):
            common_fc_out = self._common_fc(x)
            vec_out = self._vec_decoder(common_fc_out)
            img_out = self._img_decoder(common_fc_out)
            return vec_out, img_out