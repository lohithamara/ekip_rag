from api.container import ApplicationContainer

from ingestion.cleaning.factory import create_default_cleaning_pipeline
from ingestion.services.document_cleaning_service import DocumentCleaningService
from ingestion.services.ingestion_processor import IngestionProcessor
from ingestion.storage.s3_client import S3Client
from ingestion.workers.document_ingestion_worker import DocumentIngestionWorker
from ingestion.parsers.registry import ParserRegistry
from ingestion.parsers.factory import create_default_parser_registry

from rag.chunking.service import ChunkingService

from rag.citations.validator import CitationValidator

from rag.embeddings.config import EmbeddingConfig
from rag.embeddings.model import EmbeddingModel
from rag.embeddings.service import EmbeddingService

from rag.generation.config import ContextBuilderConfig
from rag.generation.context_builder import ContextBuilder
from rag.generation.grounded_answer_gate import GroundedAnswerGate

from rag.llm.config import LLMConfig
from rag.llm.factory import create_llm
from rag.llm.service import LLMService

from rag.memory.config import MemoryConfig
from rag.memory.service import ConversationMemoryService
from rag.memory.store import InMemoryConversationStore
from rag.memory.contextualization.config import ContextualizerConfig
from rag.memory.contextualization.service import ConversationContextualizationService

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

from rag.analytics.service import AnalyticsService

from rag.sql.service import SQLService

from rag.documents.service import DocumentService

from security.authorization.repository import PermissionRepository
from security.authorization.service import AuthorizationService

from database.repositories.document_repository import DatabaseDocumentRepository
from database.repositories.department_repository import DepartmentRepository
from database.repositories.user_repository import UserRepository
from database.session import SessionLocal

from rag.cache.config import RedisConfig
from rag.cache.service import RedisCacheService
from rag.cache.semantic_config import SemanticCacheConfig
from rag.cache.semantic_store import SemanticCacheStore

db = SessionLocal()

def create_container() -> ApplicationContainer:

    vector_store = QdrantVectorStore(VectorDBConfig())

    redis_cache_service = RedisCacheService(RedisConfig())

    semantic_cache_store = SemanticCacheStore(
            client=vector_store.client,
            vector_size=(
                vector_store.config.vector_size
            ),
            config=SemanticCacheConfig(),
        )
    

    document_repository = DatabaseDocumentRepository(db)
    
    department_repository = DepartmentRepository(db)

    user_repository = UserRepository(db)

    s3_client = S3Client()

    parser_registry = create_default_parser_registry()

    ingestion_processor = IngestionProcessor(
        s3_client=s3_client,
        parser_registry=parser_registry,
    )

    cleaning_pipeline = create_default_cleaning_pipeline()

    cleaning_service = DocumentCleaningService(
        cleaning_pipeline=cleaning_pipeline,
    )

    chunking_service = ChunkingService()

    embedding_model = EmbeddingModel(EmbeddingConfig())

    embedding_service = EmbeddingService(
        model=embedding_model,
        config=EmbeddingConfig(),
    )

    document_worker = (
        DocumentIngestionWorker(
            document_repository=document_repository,
            department_repository=department_repository,
            ingestion_processor=ingestion_processor,
            cleaning_service=cleaning_service,
            chunking_service=chunking_service,
            embedding_service=embedding_service,
            vector_store=vector_store,
        )
    )

    document_service = (
        DocumentService(

            repository=document_repository,

            department_repository=(
                department_repository
            ),

            user_repository=(
                user_repository
            ),

            s3_client=s3_client,

            worker=document_worker,

            vector_store=vector_store,
        )
    )

    ingestion_processor = (
        IngestionProcessor(
            s3_client=s3_client,
            parser_registry=parser_registry,
        )
    )

    analytics_service = AnalyticsService()
    
    sql_service = SQLService()

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

    context_builder = ContextBuilder(ContextBuilderConfig())

    prompt_builder = PromptBuilder(PromptConfig())

    llm_service = LLMService(
        create_llm(
            LLMConfig()
        )
    )

    # ---------------------------------
    # AGENT SERVICE
    # ---------------------------------

    authorization_service = (
        AuthorizationService(
            PermissionRepository(db)
        )
    )

    grounded_answer_gate = GroundedAnswerGate()
    # ---------------------------------
    # CONVERSATION MEMORY
    # ---------------------------------

    memory_store = InMemoryConversationStore()

    memory_service = (
        ConversationMemoryService(
            store=memory_store,
            config=MemoryConfig(),
        )
    )

    # ---------------------------------
    # QUERY CONTEXTUALIZATION
    # ---------------------------------

    contextualization_service = (
        ConversationContextualizationService(
            llm_service=llm_service,
            config=ContextualizerConfig(),
        )
    )

    # ---------------------------------
    # RAG SERVICE
    # ---------------------------------

    rag_service = RAGService(

        retrieval_service=retrieval_service,

        reranking_service=reranking_service,

        context_builder=context_builder,

        prompt_builder=prompt_builder,

        llm_service=llm_service,

        config=RAGConfig(),

        citation_validator=CitationValidator(),

        grounded_answer_gate=grounded_answer_gate,

        memory_service=memory_service,

        contextualization_service=contextualization_service,

        authorization_service=authorization_service,
        
        cache_service=redis_cache_service,

        semantic_cache_store=semantic_cache_store,
        
        embedding_service=embedding_service,
    )

    return ApplicationContainer(
        rag_service=rag_service,
        vector_store=vector_store,
        document_service=document_service,
        document_worker=document_worker,
    )