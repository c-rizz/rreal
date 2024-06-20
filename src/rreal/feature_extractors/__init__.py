from rreal.feature_extractors.feature_extractor import FeatureExtractor

feature_extractor_classes = {}

def get_feature_extractor(feature_extractor_name : str) -> type[FeatureExtractor]:
    return feature_extractor_classes[feature_extractor_name]

def register_feature_extractor_class(fe_class : type):
    feature_extractor_classes[fe_class.__name__] = fe_class


import rreal.feature_extractors.stack_vectors_feature_extractor 
    
