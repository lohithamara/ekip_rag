from rag.embeddings.config import EmbeddingConfig
from rag.embeddings.model import EmbeddingModel

from rag.retrieval.dense_service import DenseRetrievalService
from rag.retrieval.hybrid_service import HybridRetrievalService
from rag.retrieval.sparse_service import SparseRetrievalService

from rag.sparse_retrieval.bm25_index import BM25Index
from rag.sparse_retrieval.config import SparseRetrievalConfig

from rag.vector_store.config import VectorDBConfig
from rag.vector_store.qdrant_store import QdrantVectorStore

from rag.reranking.config import RerankerConfig
from rag.reranking.factory import create_reranker
from rag.reranking.schemas import RerankRequest
from rag.reranking.service import RerankingService

from rag.generation.config import ContextBuilderConfig
from rag.generation.context_builder import ContextBuilder
from rag.generation.schemas import ContextRequest

from rag.prompts.builder import PromptBuilder
from rag.prompts.config import PromptConfig

from rag.llm.config import LLMConfig
from rag.llm.factory import create_llm
from rag.llm.schemas import LLMRequest
from rag.llm.service import LLMService


def main():

    query = (
        "What are the approval limits "
        "for the finance team?"
    )

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

    hybrid_service = HybridRetrievalService(
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

    retrieved = hybrid_service.retrieve(
        query=query,
        tenant_id="tenant_1",
        department="finance",
        limit=10,
    )

    reranked = reranking_service.rerank(

        RerankRequest(
            query=query,
            results=retrieved,
            top_k=5,
        )

    )

    context = context_builder.build(

        ContextRequest(
            reranked_results=reranked.results,
        )

    )

    prompt = prompt_builder.build(
        query=query,
        context=context,
    )

    response = llm_service.generate(

        LLMRequest(

            system_prompt=prompt.system_prompt,

            user_prompt=prompt.user_prompt,

            temperature=0.0,

            max_tokens=300,
        )

    )

    print("=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(query)

    print()

    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(response.answer)

    print()

    print("=" * 80)
    print("SOURCES")
    print("=" * 80)

    for chunk in context.chunks:
        print("-", chunk.source_filename)

    vector_store.close()


if __name__ == "__main__":
    main()