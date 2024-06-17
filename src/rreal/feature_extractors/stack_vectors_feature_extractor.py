import torch.nn as nn
from abc import abstractmethod
from rreal.feature_extractors.feature_extractor import FeatureExtractor
import gymnasium as gym
from adarl.utils.ObsConverter import ObsConverter
import torch as th

class StackVectorsFeatureExtractor(FeatureExtractor):
    def __init__(self, observation_space : gym.spaces.Space):
        self._obs_converter = ObsConverter(observation_shape=observation_space)
        if self._obs_converter.has_image_part():
            raise NotImplementedError(f"Input observations contain images.")

    def extract_features(self, observation_batch):
        with th.no_grad():
            return self._obs_converter.getVectorPart(observation_batch)
    
    def encoding_size(self) -> int:
        return self._obs_converter.vector_part_size()