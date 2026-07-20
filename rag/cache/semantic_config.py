from dataclasses import dataclass


@dataclass(slots=True)
class SemanticCacheConfig:

    collection_name: str = (
        "ekip_semantic_cache"
    )

    similarity_threshold: float = 0.8

    enabled: bool = True