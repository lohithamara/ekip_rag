from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    batch_size: int = 32
    normalize_embeddings: bool = True
    device: str | None = None
    embedding_version: str = "1.0"

    def __post_init__(self):
        if not self.model_name.strip():
            raise ValueError(
                "model_name cannot be empty."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if not self.embedding_version.strip():
            raise ValueError(
                "embedding_version cannot be empty."
            )