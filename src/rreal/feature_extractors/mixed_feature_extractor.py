from __future__ import annotations
import torch.nn as nn
from rreal.feature_extractors.feature_extractor import FeatureExtractor
import gymnasium as gym
from adarl.utils.ObsConverter import ObsConverter
import torch as th
from rreal.feature_extractors import register_feature_extractor_class
from rreal.nets.Mixed_encoder import Mixed_encoder
import yaml
import adarl.utils.dbg.ggLog as ggLog
from typing_extensions import override
import zipfile
from adarl.utils.running_mean_std import RunningNormalizer
from adarl.utils.utils import get_func_input_args, check_dict_match
import hashlib
import math
import pickle
from typing import Any, Callable, List
from dataclasses import dataclass, replace

@dataclass
class MixedFeatureExtractorInitArgs:
    device : th.device
    """Torch device to use for the feature extractor."""
    encoding_size : int | None = 64
    """Size of the final encoding output. If None, it is the sum of the image and vector encoding sizes."""
    img_encoding_size : int = 64
    """Size of the internal image encoder hidden output.."""
    vec_encoding_size : int | None = 64
    """Size of the internal vector encoder hidden output. If None it is the same as the input vector size."""
    img_backbone : str = "conv_smaller"
    """Architecture of the image encoder. See :class:`rreal.nets.Image_encoder` for options."""
    vec_encoder_arch : List[int] | str = (64,64)
    """Architecture of the vector encoder. See :class:`rreal.utils.utils.build_mlp_net` for options."""
    combiner_arch : List[int] | str = (128,)
    """Architecture of the mlp that combines the image and vector encodings. See :class:`rreal.utils.utils.build_mlp_net` for options."""
    img_ensemble_size : int = 1
    """How many parallel image encoders to use. The outputs are averaged before being passed to the combiner."""
    vec_ensemble_size : int = 1
    """How many parallel vector encoders to use. The outputs are averaged before being passed to the combiner."""
    use_coord_conv : bool = True
    """Use CoordConv in the image encoder. See :class:`rreal.nets.Image_encoder` for details."""
    use_batchnorm : bool = True
    """Use BatchNorm in the image encoder. See :class:`rreal.nets.Image_encoder` for details."""
    use_weightnorm : bool = False
    """Use WeightNorm in the image  and vector encoders."""
    encoders_activation : Callable[[],th.nn.Module] = nn.LeakyReLU
    """Output activation of the image and vector encoders."""
    normalize_input_obs : bool = True
    """Use a running normalizer to normalize the vector part of the input observations. The image part is always normalized to [0,1]."""


class MixedFeatureExtractor(FeatureExtractor):
    """Feature extractor for dict observations containing both vector and image components.

    All the non-image entries of the observation dict get concatenated into a single vector
    (as in :class:`StackVectorsFeatureExtractor`), and the single image entry is kept as is.
    The two are then encoded and fused by a :class:`rreal.nets.Mixed_encoder.Mixed_encoder`.

    Differently from :class:`StackVectorsFeatureExtractor`, this extractor contains trainable
    parameters, and :meth:`extract_features` is differentiable: gradients flow back from
    whatever loss the algorithm computes on the encoding. :meth:`train_extractor` does nothing,
    the encoder is meant to be trained end-to-end through the actor/critic losses.
    """

    def __init__(self,  observation_space : gym.spaces.Space,
                        hp : MixedFeatureExtractorInitArgs):
        super().__init__()
        self._init_args = get_func_input_args(exclude=["self", "__class__"])
        self._normalize_input_obs = hp.normalize_input_obs
        self._th_device = hp.device
        self._obs_converter = ObsConverter(observation_shape=observation_space)
        self._vec_part_size = self._obs_converter.vector_part_size()
        self._img_channels, self._img_height, self._img_width = self._obs_converter.imageSizeCHW()
        self._has_image_part = self._obs_converter.has_image_part()
        self._has_vector_part = self._vec_part_size > 0

        # An encoding size of zero disables the corresponding branch in Mixed_encoder, so if
        # one of the two parts is missing we just switch its branch off.
        if not self._has_image_part:
            ggLog.warn(f"{type(self).__name__}: observation space has no image part, "
                       f"the image branch will be disabled. Consider using StackVectorsFeatureExtractor.")
            img_encoding_size = 0
        else:
            img_encoding_size = hp.img_encoding_size
        if not self._has_vector_part:
            ggLog.warn(f"{type(self).__name__}: observation space has no vector part, "
                       f"the vector branch will be disabled.")
            vec_encoding_size = 0
        else:
            if hp.vec_encoding_size is None:
                vec_encoding_size = self._vec_part_size
            else:
                vec_encoding_size = hp.vec_encoding_size
        if img_encoding_size == 0 and vec_encoding_size == 0:
            raise RuntimeError(f"Observation space contains neither an image nor a vector part: {observation_space}")
        self._img_encoding_size = img_encoding_size
        self._vec_encoding_size = vec_encoding_size
        self._encoding_size = hp.encoding_size if hp.encoding_size is not None else img_encoding_size + vec_encoding_size
        if isinstance(hp.combiner_arch, str) and hp.combiner_arch.lower().strip() == "identity":
            if self._img_encoding_size + self._vec_encoding_size != self._encoding_size:
                raise AttributeError(f"combiner_arch is 'identity', so encoding_size must be "
                                     f"img_encoding_size+vec_encoding_size = {self._img_encoding_size+self._vec_encoding_size}, "
                                     f"but it is {self._encoding_size}")

        img_pixel_low, img_pixel_high = self._obs_converter.getImgPixelRange()
        self.register_buffer("_img_pixel_low",   th.as_tensor(img_pixel_low,  dtype=th.float32, device=hp.device))
        self.register_buffer("_img_pixel_scale", th.as_tensor(1.0/(img_pixel_high-img_pixel_low),
                                                              dtype=th.float32, device=hp.device))

        if hp.normalize_input_obs and self._vec_part_size > 0:
            self._normalizer = RunningNormalizer(shape=(self._vec_part_size,),
                                                 dtype = self._obs_converter.getVectorPartDtype(),
                                                 device=self._th_device)
            self._normalizer = th.compile(self._normalizer, mode="max-autotune")
        else:
            self._normalizer = th.nn.Identity()

        self._encoder = Mixed_encoder(  image_channels = self._img_channels,
                                        image_width = self._img_width,
                                        image_height = self._img_height,
                                        img_ensemble_size = hp.img_ensemble_size,
                                        img_encoding_size = self._img_encoding_size,
                                        img_backbone = hp.img_backbone,
                                        torchDevice = hp.device,
                                        use_coord_conv = hp.use_coord_conv,
                                        vec_encoder_arch = hp.vec_encoder_arch,
                                        vec_part_size = self._vec_part_size,
                                        vec_encoding_size = self._vec_encoding_size,
                                        vec_ensemble_size = hp.vec_ensemble_size,
                                        output_size = self._encoding_size,
                                        combiner_arch = hp.combiner_arch,
                                        encoders_activation = hp.encoders_activation,
                                        use_batchnorm = hp.use_batchnorm,
                                        use_weightnorm = hp.use_weightnorm)
        self._encoder.to(hp.device)

    def _preprocess(self, observation_batch) -> tuple[th.Tensor, th.Tensor, th.Size]:
        """Extract, flatten and normalize the vector and image parts, returning them together with
        the leading batch dimensions that were flattened away. No gradient is needed here, the
        observations are constants.

        Observations may come as (batch, ...) or as (batch, trajectory, ...), while the encoder
        only accepts a single leading batch dimension. The flattening must happen *before*
        normalization, otherwise the running statistics get computed over the wrong dimensions.
        """
        with th.no_grad():
            vec_part = None
            img_part = None
            batch_dims = th.Size()
            if self._has_vector_part:
                th.cuda.nvtx.mark("get vec part")
                vec_part = self._obs_converter.getVectorPart(observation_batch)
                batch_dims = vec_part.size()[:-1]
                vec_part = vec_part.reshape(-1, self._vec_part_size)
                th.cuda.nvtx.mark("normalize")
                vec_part = self._normalizer(vec_part).clone()
            if self._has_image_part:
                th.cuda.nvtx.mark("get img part")
                img_part = self._obs_converter.getImgPart(observation_batch)
                if vec_part is None:
                    batch_dims = img_part.size()[:-3]
                img_part = img_part.reshape(-1, self._img_channels, self._img_height, self._img_width)
                img_part = (img_part.to(device=self._th_device, dtype=th.float32) - self._img_pixel_low)*self._img_pixel_scale
            # A missing part becomes an empty tensor with the right batch size: the corresponding
            # branch of Mixed_encoder is disabled and only looks at its size()[0].
            flat_batch_size = math.prod(batch_dims)
            if vec_part is None:
                vec_part = th.empty(size=(flat_batch_size,0), device=self._th_device)
            if img_part is None:
                img_part = th.empty(size=(flat_batch_size,0,0,0), device=self._th_device)
        return vec_part, img_part, batch_dims

    def extract_features(self, observation_batch) -> th.Tensor:
        vec_flat, img_flat, batch_dims = self._preprocess(observation_batch)
        th.cuda.nvtx.mark("encode")
        encoding = self._encoder(img_flat, vec_flat)
        return encoding.reshape(batch_dims + (self._encoding_size,))

    def encoding_size(self) -> int:
        return self._encoding_size

    @override
    @classmethod
    def load(cls, file : zipfile.ZipFile | str, name : str = "feature_extractor", device : th.device | None = None) -> MixedFeatureExtractor:
        if isinstance(file,zipfile.ZipFile):
            with file.open(f"{name}.extra.fe.yaml", "r") as init_args_yamlfile:
                extra = yaml.load(init_args_yamlfile, Loader=yaml.CLoader)
            with file.open(f"{name}.state.fe.pth", "r") as state_file:
                state_dict = th.load(state_file)
        else:
            raise RuntimeError(f"Unexpected input type")
        if "class_name" in extra and extra["class_name"] != cls.__name__:
            raise RuntimeError(f"File was not saved by this class found '{extra['class_name']}' instead of '{cls.__name__}'")
        if device is not None:
            extra["init_args"]["hp"].device=device
        fe = MixedFeatureExtractor(**extra["init_args"])
        fe.load_state_dict(_adapt_compiled_state_dict(state_dict, fe))
        return fe

    @override
    def save_to_archive(self, archive : zipfile.ZipFile, name : str = "feature_extractor"):
        extra = {}
        extra["init_args"] = self._init_args
        extra["class_name"] = self.__class__.__name__
        extra["hashes"] = get_model_hashes(self)
        with archive.open(f"{name}.extra.fe.yaml", "w") as init_args_yamlfile:
            init_args_yamlfile.write(yaml.dump(extra,default_flow_style=None).encode("utf-8"))
        with archive.open(f"{name}.state.fe.pth", "w") as state_file:
            th.save(self.state_dict(), state_file)

    def train_extractor(self, global_step, grad_steps, buffer):
        # The encoder is trained end-to-end through the actor/critic losses.
        pass

    @override
    def check_init_args_match(self : FeatureExtractor, loaded_fe_name : str, loaded_fe_args : dict[str, Any]):
        if self.__class__.__name__ != loaded_fe_name:
            ggLog.warn(f"feature_extractor_class_name of loaded model differs from that of self.\n"
                       f"loaded = {loaded_fe_name}, self's = {self.__class__.__name__}")
            raise RuntimeError("Unmatched init_args")
        current_fe_args_ = self.get_init_args().copy()
        loaded_fe_args_ = loaded_fe_args.copy()
        current_fe_args_["observation_space"].seed(0) # ignore the rng state
        loaded_fe_args_["observation_space"].seed(0) # ignore the rng state
        # Ignore the device. copy() above is shallow, so the hparams must be replaced by a
        # modified copy: assigning to hp.device would corrupt the live init args of both models.
        current_fe_args_["hp"] = replace(current_fe_args_["hp"], device=None)
        loaded_fe_args_["hp"] = replace(loaded_fe_args_["hp"], device=None)
        check_dict_match(current_fe_args_, loaded_fe_args_)

register_feature_extractor_class(MixedFeatureExtractor)


def _adapt_compiled_state_dict(state_dict: dict, model: nn.Module) -> dict:
    """Handle mismatch between state dicts saved with/without torch.compile().
    torch.compile() wraps submodules under '_orig_mod', so saved keys may have
    '._orig_mod.' in them while the current model does not, or vice versa."""
    stripped = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
    model_keys = set(model.state_dict().keys())
    if any("._orig_mod." in k for k in model_keys):
        stripped_to_model = {k.replace("._orig_mod.", "."): k for k in model_keys}
        return {stripped_to_model.get(k, k): v for k, v in stripped.items()}
    return stripped


def get_model_hashes(model : nn.Module):
    return {n:hashlib.sha256(pickle.dumps(v.cpu().numpy().tobytes())).hexdigest() for n,v in model.state_dict().items()}
