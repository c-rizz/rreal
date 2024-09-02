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
class StackVectorsFeatureExtractor(FeatureExtractor):
    def __init__(self, observation_space : gym.spaces.Space):
        super().__init__()
        _, _, _, values = inspect.getargvalues(inspect.currentframe())
        self._init_args = values
        self._init_args.pop("self")
        self._init_args.pop("__class__")
        self._obs_converter = ObsConverter(observation_shape=observation_space)
        if self._obs_converter.has_image_part():
            raise NotImplementedError(f"Input observations contain images.")

    def extract_features(self, observation_batch):
        with th.no_grad():
            return self._obs_converter.getVectorPart(observation_batch)
    
    def encoding_size(self) -> int:
        return self._obs_converter.vector_part_size()
    
    
    @classmethod
    def load(cls, path : str):
        with open(path+".extra.yaml", "r") as init_args_yamlfile:
            extra = yaml.load(init_args_yamlfile, Loader=yaml.CLoader)
        if "class_name" in extra and extra["class_name"] != cls.__name__:
            raise RuntimeError(f"File was not saved by this class")
        return StackVectorsFeatureExtractor(**extra["init_args"])
    
    def save(self, path : str):
        extra = {}
        extra["init_args"] = self._init_args
        extra["class_name"] = self.__class__.__name__
        # ggLog.info(f"saving extra={extra}")
        with open(path+".extra.yaml", "w") as init_args_yamlfile:
            yaml.dump(extra,init_args_yamlfile, default_flow_style=None)

    def train(self, global_step, grad_steps, buffer):
        pass
    

register_feature_extractor_class(StackVectorsFeatureExtractor)