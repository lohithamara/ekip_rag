from sentence_transformers import SentenceTransformer

from rag.embeddings.config import EmbeddingConfig


class EmbeddingModel:

    def __init__(
        self,
        config: EmbeddingConfig,
    ):
        self.config = config

        self.model = SentenceTransformer(
            config.model_name,
            device=config.device,
        )

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def encode(
        self,
        texts: list[str],
    ) -> list[tuple[float, ...]]:

        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=(
                self.config.normalize_embeddings
            ),
            show_progress_bar=False,
        )

        return [
            tuple(
                float(value)
                for value in vector
            )
            for vector in vectors
        ]