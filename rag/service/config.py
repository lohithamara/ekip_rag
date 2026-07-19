from dataclasses import dataclass


@dataclass(slots=True)
class RAGConfig:

    # Retrieval
    retrieval_limit: int = 10

    # Reranking
    rerank_top_k: int = 5

    # LLM
    temperature: float = 0.0

    max_tokens: int = 512