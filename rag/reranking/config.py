from dataclasses import dataclass, field


@dataclass(slots=True)
class RerankerConfig:
    """
    Configuration for the reranking module.
    """

    # Hugging Face model
    model_name: str = (
        "BAAI/bge-reranker-base"
    )

    # Device
    device: str = "cpu"

    # Batch size for scoring
    batch_size: int = 16

    # Number of chunks passed to the reranker
    candidate_limit: int = 10

    # Number of chunks returned after reranking
    top_k: int = 5

    # Trust remote code (for some HF models)
    trust_remote_code: bool = False

    # Extra model-specific settings
    extra: dict = field(
        default_factory=dict
    )