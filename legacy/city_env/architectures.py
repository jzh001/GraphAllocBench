from city_env.feature_extractors.mlp_extractor import MLPFeatureExtractor
from city_env.feature_extractors.mlp_extractor_large import MLPFeatureExtractorLarge
from city_env.feature_extractors.bipartite_gnn_extractor import BipartiteGNNExtractor
from city_env.feature_extractors.hybrid_extractor import HybridGNNExtractor
from city_env.feature_extractors.hgnn_extractor import HGNNExtractor
from city_env.feature_extractors.hgnn_lora import HGNNLoRAExtractor
from city_env.feature_extractors.hgnn_sparse_extractor import HGNNSparseExtractor

architectures = [
    # 0
    dict(
        features_extractor_class=MLPFeatureExtractor,
        features_extractor_kwargs=dict(embedding_dim=128),
        net_arch=dict(pi=[256, 256], vf=[256, 256, 128])
    ),
    # 1.
    dict(
        features_extractor_class=MLPFeatureExtractorLarge,
        features_extractor_kwargs=dict(embedding_dim=128),
        net_arch=dict(pi=[256, 256], vf=[256, 256, 128])
    ),
    # 2.
    dict(
        features_extractor_class=HGNNSparseExtractor,
        features_extractor_kwargs=dict(features_dim=128, pooling_types = ['mean', 'max'], pooling_strategy='concat'),
        net_arch=dict(pi=[256, 256], vf=[256, 256, 128])
    ),
    # 3.
    dict(
        features_extractor_class=HGNNSparseExtractor,
        features_extractor_kwargs=dict(features_dim=128, pooling_types = ['multi_head_attention'], pooling_strategy='concat'),
        net_arch=dict(pi=[256, 256], vf=[256, 256, 128])
    ),
]