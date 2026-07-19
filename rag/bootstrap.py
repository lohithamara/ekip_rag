from rag.embeddings.config import EmbeddingConfig
from rag.embeddings.model import EmbeddingModel

from rag.generation.config import ContextBuilderConfig
from rag.generation.context_builder import ContextBuilder

from rag.llm.config import LLMConfig
from rag.llm.factory import create_llm
from rag.llm.service import LLMService

from rag.prompts.builder import PromptBuilder
from rag.prompts.config import PromptConfig

from rag.reranking.config import RerankerConfig
from rag.reranking.factory import create_reranker
from rag.reranking.service import RerankingService

from rag.retrieval.dense_service import DenseRetrievalService
from rag.retrieval.hybrid_service import HybridRetrievalService
from rag.retrieval.sparse_service import SparseRetrievalService

from rag.service.config import RAGConfig
from rag.service.service import RAGService

from rag.sparse_retrieval.bm25_index import BM25Index
from rag.sparse_retrieval.config import SparseRetrievalConfig

from rag.vector_store.config import VectorDBConfig
from rag.vector_store.qdrant_store import QdrantVectorStore


def create_rag_service() -> tuple[RAGService, QdrantVectorStore]:

    vector_store = QdrantVectorStore(
        VectorDBConfig()
    )

    embedding_model = EmbeddingModel(
        EmbeddingConfig()
    )

    dense_service = DenseRetrievalService(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    sparse_service = SparseRetrievalService(
        BM25Index.load(
            SparseRetrievalConfig()
        )
    )

    retrieval_service = HybridRetrievalService(
        dense_service=dense_service,
        sparse_service=sparse_service,
    )

    reranking_service = RerankingService(
        create_reranker(
            RerankerConfig()
        )
    )

    context_builder = ContextBuilder(
        ContextBuilderConfig()
    )

    prompt_builder = PromptBuilder(
        PromptConfig()
    )

    llm_service = LLMService(
        create_llm(
            LLMConfig()
        )
    )

    rag_service = RAGService(
        retrieval_service=retrieval_service,
        reranking_service=reranking_service,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm_service=llm_service,
        config=RAGConfig(),
    )

    return rag_service, vector_store