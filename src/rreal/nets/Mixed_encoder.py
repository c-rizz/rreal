from adarl.utils.dbg.dbg_checks import dbg_check_size
from rreal.nets.Image_encoder import Image_encoder
from rreal.nets.Parallel import Parallel

import torch as th
import torch.nn as nn
from typing import Tuple, List, Callable

import adarl.utils.dbg.ggLog as ggLog
from rreal.utils.utils import build_mlp_net, scale_layer_weights, softclamp

class Mixed_VAE_encoder(nn.Module):
    def __init__(self,  image_channels : int = 1,
                        image_width : int = 64,
                        image_height : int = 64,
                        img_ensemble_size = 1,
                        img_encoding_size = 0,
                        backbone : str = "conv",
                        checkDimensions : bool = True,
                        torchDevice : str | th.device = "cuda",
                        use_coord_conv : bool = True,
                        mu_activation_class = th.nn.Identity,
                        dropout_prob : float = 0,
                        vec_encoder_arch : List[int] | str = [64,64],
                        vec_part_size = 0,
                        vec_encoding_size = 0,
                        vec_ensemble_size = 1,
                        latent_space_size = 32,
                        combiner_arch = [128],
                        fcs_arch = [64],
                        encoders_activation = th.nn.LeakyReLU,
                        use_batchnorm = True,
                        use_weightnorm : bool = False,
                        concatenate_mulogvar : bool= False,
                        max_logvar : float = 20.0):
        super().__init__()
        self._checkDimensions = checkDimensions
        self._latent_space_size = latent_space_size
        self._mu_activation_class = mu_activation_class
        self._fcs_arch = fcs_arch
        self._fcs_ensemble_size = 1
        self._concatenate_mulogvar = concatenate_mulogvar
        self._max_logvar = max_logvar

        if dropout_prob > 0:
            raise NotImplementedError("Dropout is not implemented yet for Mixed_VAE_encoder. Please set dropout_prob=0.")

        if type(combiner_arch)==str and combiner_arch.lower().strip() == "identity":
            combined_size = img_encoding_size + vec_encoding_size
        else:
            combined_size = self._latent_space_size*2
        self.encoder = Mixed_encoder(  image_channels = image_channels,
                                        image_width = image_width,
                                        image_height = image_height,
                                        img_ensemble_size = img_ensemble_size,
                                        img_encoding_size = img_encoding_size,
                                        img_backbone = backbone,
                                        torchDevice = torchDevice,
                                        use_coord_conv = use_coord_conv,
                                        vec_encoder_arch = vec_encoder_arch,
                                        vec_part_size = vec_part_size,
                                        vec_encoding_size = vec_encoding_size,
                                        vec_ensemble_size = vec_ensemble_size,
                                        output_size = combined_size,
                                        combiner_arch = combiner_arch,
                                        encoders_activation = encoders_activation,
                                        use_batchnorm = use_batchnorm,
                                        use_weightnorm = use_weightnorm)
        self.fc_mu = build_mlp_net( self._fcs_arch, 
                                    input_size=combined_size,
                                    output_size=self._latent_space_size,
                                    last_activation_class=self._mu_activation_class,
                                    return_ensemble_mean=True,
                                    ensemble_size=self._fcs_ensemble_size,
                                    use_weightnorm = use_weightnorm)
        self.fc_logvar = build_mlp_net( self._fcs_arch, 
                                        input_size=combined_size,
                                        output_size=self._latent_space_size,
                                        last_activation_class= th.nn.Identity, #lambda: ScaledTanh(50),
                                        return_ensemble_mean=True,
                                        ensemble_size=self._fcs_ensemble_size,
                                        use_weightnorm = use_weightnorm,
                                        last_layer_init_func=lambda m: scale_layer_weights(m, multiplier=0.0001))  # see https://stackoverflow.com/questions/49634488/keras-variational-autoencoder-nan-loss

    @property
    def latent_space_size(self):
        return self._latent_space_size

    def input_img_width(self):
        return self.encoder.input_img_width()

    def input_img_height(self):
        return self.encoder.input_img_height()

    def input_img_channels(self):
        return self.encoder.input_img_channels()

    def input_vec_size(self):
        return self.encoder.input_vec_size()
    
    def forward(self, image: th.Tensor, vector: th.Tensor) -> Tuple[th.Tensor, th.Tensor] | th.Tensor:

        batch_size = image.size()[0]

        combined_mixed_encoding = self.encoder(image, vector)

        mu = self.fc_mu(combined_mixed_encoding)
        logvar = self.fc_logvar(combined_mixed_encoding)
        logvar = softclamp(logvar, maxval=self._max_logvar)

        if self._checkDimensions:
            assert mu.size() == (batch_size, self._latent_space_size)
            assert logvar.size() == (batch_size, self._latent_space_size)

        if self._concatenate_mulogvar:
            return th.cat([mu, logvar], dim=1)
        else:
            return mu, logvar
            
    def sample(self, mu : th.Tensor, logvar : th.Tensor):
        std = th.exp(0.5 * logvar) # std = sqrt(var) = sqrt(e^logvar) = e^(0.5*logvar)
        eps = th.randn(std.size(), device=mu.device) # sample from unit gaussian
        return eps * std + mu


class Mixed_encoder(nn.Module):
    def __init__(self,  image_channels : int = 1,
                        image_width : int = 64,
                        image_height : int = 64,
                        img_ensemble_size = 1,
                        img_encoding_size = 0,
                        img_backbone : str | None = "conv",
                        torchDevice : str | th.device = "cuda",
                        use_coord_conv : bool = True,
                        vec_encoder_arch : List[int] | str | None = [64,64],
                        vec_part_size = 0,
                        vec_encoding_size = 0,
                        vec_ensemble_size = 1,
                        output_size = 32,
                        combiner_arch = [128],
                        encoders_activation : Callable[[],th.nn.Module] = th.nn.LeakyReLU,
                        use_batchnorm = True,
                        use_weightnorm : bool = False,
                        vec_layers_init_func : Callable[[nn.Module], None] | None = None,
                        last_layer_init_func : Callable[[nn.Module], None] | None = None):
        super().__init__()
        self._input_width  = image_width
        self._input_height = image_height
        self._input_channels = image_channels
        self._img_encoding_size = img_encoding_size
        self._latent_space_size = output_size
        self._ensemble_size = img_ensemble_size
        self._img_ensemble_size = img_ensemble_size
        self._vec_enc_ensemble_size = vec_ensemble_size
        self._vec_encoding_size = vec_encoding_size
        self._vec_part_size = vec_part_size
        self._vec_encoder_arch = vec_encoder_arch
        self._combiner_arch = combiner_arch
        self._fcs_ensemble_size = 1
        self._encoders_activation = encoders_activation

        if self._img_encoding_size != 0:
            self._backbone = img_backbone.lower()
            # if backbone == "mobilenetv3" or backbone == "resnet18" or backbone == "bigconv":
            #     self._conv_ensemble_size = 1
            self.img_encoder = Parallel([Image_encoder(   image_channels_num = image_channels,
                                                    net_input_width = image_width,
                                                    net_input_height = image_height,
                                                    backbone = img_backbone,
                                                    torchDevice = torchDevice,
                                                    use_coord_conv = use_coord_conv,
                                                    last_activation_class=self._encoders_activation,
                                                    output_size=self._img_encoding_size,
                                                    use_batchnorm = use_batchnorm,
                                                    use_weightnorm = use_weightnorm,
                                                    fc_net_arch = [])
                                        for _ in range(self._img_ensemble_size)],
                                        return_mean=True)
        else:
            self.img_encoder = lambda img: th.empty(size = (img.size()[0],0), device=torchDevice)
        
        if self._vec_encoding_size != 0:
            self.vec_encoder = build_mlp_net(  self._vec_encoder_arch, 
                                                input_size=self._vec_part_size,
                                                output_size=self._vec_encoding_size,
                                                last_activation_class=self._encoders_activation,
                                                return_ensemble_mean=True,
                                                ensemble_size=self._vec_enc_ensemble_size,
                                                use_weightnorm = use_weightnorm,
                                                layer_init_func = vec_layers_init_func)
        else:
            self.vec_encoder = lambda vec: th.empty(size = (vec.size()[0],0), device=torchDevice)

        self.combiner = build_mlp_net(  self._combiner_arch, 
                                        input_size=self._img_encoding_size + self._vec_encoding_size,
                                        output_size=output_size,
                                        last_activation_class=th.nn.LeakyReLU,
                                        return_ensemble_mean=True,
                                        ensemble_size=1,
                                        use_weightnorm = use_weightnorm,
                                        layer_init_func = vec_layers_init_func,
                                        last_layer_init_func=last_layer_init_func)
        
        

        

    @property
    def latent_space_size(self):
        return self._latent_space_size

    def input_img_width(self):
        return self._input_width

    def input_img_height(self):
        return self._input_height

    def input_img_channels(self):
        return self._input_channels

    def input_vec_size(self):
        return self._vec_part_size
    
    def forward(self, image: th.Tensor, vector: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:

        batch_size = image.size()[0]
        dbg_check_size(image, (batch_size,self._input_channels,self._input_height, self._input_width), "Mixed_encoder.forward: image")
        dbg_check_size(vector, (batch_size,self._vec_part_size), "Mixed_encoder.forward: vector")

        # ggLog.info(f"Mixed_encoder.forward: image.size()= {image.size()}, vector.size()= {vector.size()}")
        img_encoding = self.img_encoder(image)
        dbg_check_size(img_encoding, (batch_size,self._img_encoding_size), "Mixed_encoder.forward: img_encoding")
        vec_encoding = self.vec_encoder(vector)
        dbg_check_size(vec_encoding, (batch_size,self._vec_encoding_size), "Mixed_encoder.forward: vec_encoding")

        # enc_batch = th.cat([img_encoding, vec_encoding], dim = 1)
        # ggLog.info(f"enc_batch.size()= {enc_batch.size()}")
        return self.combiner(th.cat([img_encoding, vec_encoding], dim = 1))

class DictMixedEncoder(Mixed_encoder):
    def __init__(self, image_dict_key : str | int,
                       vector_dict_key : str | int,
                       **mixed_encoder_kwargs):
        self._image_dict_key = image_dict_key
        self._vector_dict_key = vector_dict_key
        super().__init__(**mixed_encoder_kwargs)

    def forward(self, dict_obs : dict[str | int, th.Tensor]) -> th.Tensor:
        image = dict_obs[self._image_dict_key]
        vector = dict_obs[self._vector_dict_key]
        return super().forward(image, vector)