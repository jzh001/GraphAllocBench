from .feature_extractors.mlp_extractor import MLPFeatureExtractor
from .feature_extractors.mlp_extractor_large import MLPFeatureExtractorLarge
from .feature_extractors.hgnn_sparse_extractor import HGNNSparseExtractor

architectures = [
    dict(
        features_extractor_class=MLPFeatureExtractor,
        features_extractor_kwargs=dict(embedding_dim=128),
        net_arch=dict(pi=[256, 256], vf=[256, 256, 128])
    ),
    dict(
        features_extractor_class=MLPFeatureExtractorLarge,
        features_extractor_kwargs=dict(embedding_dim=128),
        net_arch=dict(pi=[256, 256], vf=[256, 256, 128])
    ),
    dict(
        features_extractor_class=HGNNSparseExtractor,
        features_extractor_kwargs=dict(features_dim=128, pooling_types = ['mean', 'max'], pooling_strategy='concat'),
        net_arch=dict(pi=[256, 256], vf=[256, 256, 128])
    ),
    dict(
        features_extractor_class=HGNNSparseExtractor,
        features_extractor_kwargs=dict(features_dim=128, pooling_types = ['multi_head_attention'], pooling_strategy='concat'),
        net_arch=dict(pi=[256, 256], vf=[256, 256, 128])
    ),
]

__all__ = ["architectures"]
