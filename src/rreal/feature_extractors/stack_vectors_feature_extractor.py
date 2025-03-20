from __future__ import annotations
import torch.nn as nn
from abc import abstractmethod
from rreal.feature_extractors.feature_extractor import FeatureExtractor
import gymnasium as gym
from adarl.utils.ObsConverter import ObsConverter
import torch as th
from rreal.feature_extractors import register_feature_extractor_class
import inspect
import yaml
import adarl.utils.dbg.ggLog as ggLog
from typing_extensions import override
import zipfile
from adarl.utils.running_mean_std import RunningNormalizer

class StackVectorsFeatureExtractor(FeatureExtractor):
    def __init__(self,  observation_space : gym.spaces.Space,
                        device : th.device,
                        normalize_input_obs : bool = True):
        super().__init__()
        _, _, _, values = inspect.getargvalues(inspect.currentframe())
        self._init_args = values
        self._init_args.pop("self")
        self._init_args.pop("__class__")
        self._normalize_input_obs = normalize_input_obs
        self._th_device = device
        self._obs_converter = ObsConverter(observation_shape=observation_space)
        if normalize_input_obs:
            self._normalizer = RunningNormalizer(shape=(self._obs_converter.vector_part_size(),),
                                                dtype = self._obs_converter.getVectorPartDtype(),
                                                device=self._th_device)
        if self._obs_converter.has_image_part():
            raise NotImplementedError(f"Input observations contain images.")

    def extract_features(self, observation_batch) -> th.Tensor:
        with th.no_grad():
            vec_part = self._obs_converter.getVectorPart(observation_batch)
            return self._normalizer(vec_part)
    
    def encoding_size(self) -> int:
        return self._obs_converter.vector_part_size()
    
    
    @override
    @classmethod
    def load(cls, file : zipfile.ZipFile | str):
        # if isinstance(file,str): # just for compatibility
        #     fname = file+".feature_extractor.extra.yaml"
        #     ggLog.info(f"opening {fname}")
        #     with open(fname, "r") as init_args_yamlfile:
        #         extra = yaml.load(init_args_yamlfile, Loader=yaml.CLoader)
        #     with open(file+"feature_extractor.state.pth", "r") as state_file:
        #         state_dict = th.load(state_file)
        # elif isinstance(file,zipfile.ZipFile):
        if isinstance(file,zipfile.ZipFile):
            with file.open("feature_extractor.extra.yaml", "r") as init_args_yamlfile:
                extra = yaml.load(init_args_yamlfile, Loader=yaml.CLoader)
            with file.open("feature_extractor.state.pth", "r") as state_file:
                state_dict = th.load(state_file)
        else:
            raise RuntimeError(f"Unexpected input type")
        if "class_name" in extra and extra["class_name"] != cls.__name__:
            raise RuntimeError(f"File was not saved by this class found '{extra['class_name']}' instead of '{cls.__name__}'")
        fe = StackVectorsFeatureExtractor(**extra["init_args"])
        fe.load_state_dict(state_dict)
    
    @override
    def save_to_archive(self, archive : zipfile.ZipFile):
        extra = {}
        extra["init_args"] = self._init_args
        extra["class_name"] = self.__class__.__name__
        # ggLog.info(f"saving extra={extra}")
        with archive.open("feature_extractor.extra.yaml", "w") as init_args_yamlfile:
            init_args_yamlfile.write(yaml.dump(extra,default_flow_style=None).encode("utf-8"))
        with archive.open("feature_extractor.state.pth", "w") as state_file:
            th.save(self.state_dict(), state_file)
            

    def train_extractor(self, global_step, grad_steps, buffer):
        pass
    

register_feature_extractor_class(StackVectorsFeatureExtractor)