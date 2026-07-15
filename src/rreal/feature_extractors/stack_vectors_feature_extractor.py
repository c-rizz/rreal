from __future__ import annotations
import torch.nn as nn
from abc import abstractmethod
from rreal.feature_extractors.feature_extractor import FeatureExtractor
import gymnasium as gym
from adarl.utils.ObsConverter import ObsConverter
import torch as th
from rreal.feature_extractors import register_feature_extractor_class
import yaml
import adarl.utils.dbg.ggLog as ggLog
from typing_extensions import override
import zipfile
from adarl.utils.running_mean_std import RunningNormalizer
from adarl.utils.utils import get_func_input_args, check_dict_match
import hashlib
import copy
import pickle
from typing import Any

class StackVectorsFeatureExtractor(FeatureExtractor):
    def __init__(self,  observation_space : gym.spaces.Space,
                        device : th.device,
                        normalize_input_obs : bool = True):
        super().__init__()
        self._init_args = get_func_input_args(exclude=["self", "__class__"])
        self._normalize_input_obs = normalize_input_obs
        self._th_device = device
        self._obs_converter = ObsConverter(observation_shape=observation_space)
        if normalize_input_obs:
            self._normalizer = RunningNormalizer(shape=(self._obs_converter.vector_part_size(),),
                                                dtype = self._obs_converter.getVectorPartDtype(),
                                                device=self._th_device)
            self._normalizer = th.compile(self._normalizer, mode="max-autotune")
        if self._obs_converter.has_image_part():
            raise NotImplementedError(f"Input observations contains non-monodimensional tensors (images?).")

    def extract_features(self, observation_batch) -> th.Tensor:
        with th.no_grad():
            th.cuda.nvtx.mark("get vec part")
            vec_part = self._obs_converter.getVectorPart(observation_batch)
            # ggLog.info(f"vec_part.devices = {vec_part.device}")
            th.cuda.nvtx.mark("normalize")
            return self._normalizer(vec_part).clone()
    
    def encoding_size(self) -> int:
        return self._obs_converter.vector_part_size()
    
    
    @override
    @classmethod
    def load(cls, file : zipfile.ZipFile | str, name : str = "feature_extractor"):
        # if isinstance(file,str): # just for compatibility
        #     fname = file+".feature_extractor.extra.yaml"
        #     ggLog.info(f"opening {fname}")
        #     with open(fname, "r") as init_args_yamlfile:
        #         extra = yaml.load(init_args_yamlfile, Loader=yaml.CLoader)
        #     with open(file+"feature_extractor.state.pth", "r") as state_file:
        #         state_dict = th.load(state_file)
        # elif isinstance(file,zipfile.ZipFile):
        if isinstance(file,zipfile.ZipFile):
            with file.open(f"{name}.extra.fe.yaml", "r") as init_args_yamlfile:
                extra = yaml.load(init_args_yamlfile, Loader=yaml.CLoader)
            with file.open(f"{name}.state.fe.pth", "r") as state_file:
                state_dict = th.load(state_file)
        else:
            raise RuntimeError(f"Unexpected input type")
        if "class_name" in extra and extra["class_name"] != cls.__name__:
            raise RuntimeError(f"File was not saved by this class found '{extra['class_name']}' instead of '{cls.__name__}'")
        fe = StackVectorsFeatureExtractor(**extra["init_args"])
        fe.load_state_dict(_adapt_compiled_state_dict(state_dict, fe))
    
    @override
    def save_to_archive(self, archive : zipfile.ZipFile, name : str = "feature_extractor"):
        extra = {}
        extra["init_args"] = self._init_args
        extra["class_name"] = self.__class__.__name__
        extra["hashes"] = get_model_hashes(self)
        # ggLog.info(f"saving extra={extra}")
        with archive.open(f"{name}.extra.fe.yaml", "w") as init_args_yamlfile:
            init_args_yamlfile.write(yaml.dump(extra,default_flow_style=None).encode("utf-8"))
        with archive.open(f"{name}.state.fe.pth", "w") as state_file:
            th.save(self.state_dict(), state_file)
            

    def train_extractor(self, global_step, grad_steps, buffer):
        pass
    
    @override
    def check_init_args_match(self : FeatureExtractor, loaded_fe_name : str, loaded_fe_args : dict[str, Any]):
        if self.__class__.__name__ != loaded_fe_name:
            ggLog.warn(f"feature_extractor_class_name of loaded model differs from that of self.\n"
                       f"loaded = {loaded_fe_name}, self's = {self._critic_feature_extractor.__class__.__name__}")
            raise RuntimeError("Unmatched init_args")
        current_fe_args_ = self.get_init_args().copy()
        loaded_fe_args_ = loaded_fe_args.copy()
        current_fe_args_["observation_space"].seed(0) # ignore the rng state
        loaded_fe_args_["observation_space"].seed(0) # ignore the rng state
        loaded_fe_args_["device"] = None # ignore the device
        current_fe_args_["device"] = None # ignore the device
        check_dict_match(current_fe_args_, loaded_fe_args_)

register_feature_extractor_class(StackVectorsFeatureExtractor)


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