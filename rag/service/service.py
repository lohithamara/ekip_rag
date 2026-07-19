from rag.generation.schemas import ContextRequest

from rag.llm.schemas import LLMRequest

from rag.memory.contextualization.schemas import ContextualizationRequest

from rag.memory.schemas import MemoryRequest

from rag.prompts.schemas import PromptRequest

from rag.reranking.schemas import RerankRequest

from rag.service.schemas import (
    RAGRequest,
    RAGResponse,
)

from rag.evaluation.trace import RAGTrace

from rag.generation.grounded_answer_gate import GroundedAnswerGate

from rag.retrieval.schemas import RetrievalRequest

from security.authorization.schemas import DepartmentScope

from security.authorization.service import AuthorizationService

from uuid import uuid4

class RAGService:

    def __init__(
        self,
        retrieval_service,
        reranking_service,
        context_builder,
        prompt_builder,
        llm_service,
        config,
        citation_validator,
        grounded_answer_gate:
        GroundedAnswerGate,
        memory_service,
        contextualization_service,
        authorization_service,
        cache_service,
        semantic_cache_store,
        embedding_service,
        trace_enabled: bool = False,
    ):
        self.retrieval_service = retrieval_service

        self.reranking_service = reranking_service

        self.context_builder = context_builder

        self.prompt_builder = prompt_builder

        self.llm_service = llm_service

        self.config = config

        self.citation_validator = citation_validator

        self.grounded_answer_gate = grounded_answer_gate

        self.memory_service = memory_service

        self.contextualization_service = contextualization_service

        self.authorization_service = authorization_service
        
        self.cache_service = cache_service

        self.semantic_cache_store = semantic_cache_store

        self.embedding_service = embedding_service

        self.trace_enabled = trace_enabled

        self.last_trace = None

    def answer(
        self,
        request: RAGRequest,
    ) -> RAGResponse:

        # --------------------------------------------------
        # 1. LOAD CONVERSATION MEMORY
        # --------------------------------------------------

        memory = self._load_memory(request)

        # --------------------------------------------------
        # 2. CONTEXTUALIZE RETRIEVAL QUERY
        # --------------------------------------------------

        contextualization = self._contextualize(request,memory,)

        retrieval_query = contextualization.contextualized_query

        effective_query = retrieval_query

        # print("QUERY")
        # print(request.query)

        # print("=" * 80)

        scope = self._authorize(request,)

        # --------------------------------------------------
        # 3. SEMANTIC CACHE LOOKUP
        # --------------------------------------------------

        query_vector = self.embedding_service.embed_query(effective_query)

        cache_match = (
            self.semantic_cache_store.search(
                vector=query_vector,
                tenant_id=str(
                    request.user.tenant_id
                ),
                authorized_departments=(
                    scope.departments
                ),
            )
        )

        if cache_match is not None:

            cache_id = cache_match.payload.get("cache_id")

            if cache_id:

                cached = self.cache_service.get_response(cache_id)

                if cached is not None:

                    print("SEMANTIC CACHE HIT")

                    final_answer = cached["answer"]

                    final_sources = cached.get("sources",[],)

                    # Store this interaction
                    # in conversation memory.
                    self.memory_service.add_turn(
                        tenant_id=request.user.tenant_id,
                        conversation_id=request.conversation_id,
                        user_message=request.query,
                        assistant_message=final_answer,
                    )

                    return RAGResponse(
                        answer=final_answer,
                        sources=final_sources,
                        metadata={
                            **cached.get(
                                "metadata",
                                {},
                            ),
                            "cache_hit": True,
                            "cache_type": (
                                "semantic"
                            ),
                        },
                    )


        print("SEMANTIC CACHE MISS")


        # --------------------------------------------------
        # 3. RETRIEVAL
        # --------------------------------------------------

        retrieved = (
            self.retrieval_service.retrieve(

                RetrievalRequest(

                    query=effective_query,

                    scope=scope,

                    limit=(
                        self.config
                        .retrieval_limit
                    ),
                )
            )
        )

        # --------------------------------------------------
        # 4. RERANKING
        # --------------------------------------------------

        reranked = self.reranking_service.rerank(
                RerankRequest(
                    query=effective_query,
                    results=retrieved,
                    top_k=(
                        self.config
                        .rerank_top_k
                    ),
                )
            )
            
        # print("\n" + "=" * 80)
        # print("RERANKED RESULTS")
        # print("=" * 80)

        # for i, result in enumerate(reranked.results, start=1):

        #     retrieval = result.retrieval_result

        #     print(f"Rank: {i}")
        #     print("Score:", result.rerank_score)
        #     print("Source:", retrieval.metadata.get("source_filename"))
        #     print("Token Count:", retrieval.metadata.get("token_count"))
        #     print("Text Preview:")
        #     print(retrieval.text[:300])
        #     print("-" * 80)
        # --------------------------------------------------
        # 5. CONTEXT BUILDING
        # --------------------------------------------------

        context = self.context_builder.build(
                ContextRequest(
                    query=effective_query,
                    reranked_results=(
                        reranked.results
                    ),
                )
            )
            
        # print("\n" + "=" * 80)
        # print("FINAL CONTEXT")
        # print("=" * 80)

        # for chunk in context.chunks:
        #     print("Source:", chunk.source_filename)
        #     print("Tokens:", chunk.token_count)
        #     print(chunk.text[:300])
        #     print("-" * 80)

        # print("Total Context Tokens:", context.total_tokens)
        # # --------------------------------------------------
        # 6. FINAL PROMPT
        #
        # ORIGINAL QUERY + MEMORY + CONTEXT
        # --------------------------------------------------

        prompt = self.prompt_builder.build(
                PromptRequest(
                    query=request.query,
                    context=context,
                    memory=memory,
                )
            )
            

        # --------------------------------------------------
        # 7. GENERATION
        # --------------------------------------------------

        llm_response = self.llm_service.generate(
            LLMRequest(
                system_prompt=prompt.system_prompt,
                user_prompt=prompt.user_prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                )
            )
            
        

        original_answer = llm_response.answer.strip()

        # --------------------------------------------------
        # 8. CITATION VALIDATION
        # --------------------------------------------------

        allowed_sources = {
            chunk.source_filename
            for chunk in context.chunks
            if chunk.source_filename
        }

        validated_sources = self.citation_validator.validate(
                answer=original_answer,
                allowed_sources=(
                    allowed_sources
                ),
            )
            

        # --------------------------------------------------
        # 9. GROUNDED ANSWER GATE
        # --------------------------------------------------

        final_answer = self.grounded_answer_gate.apply(
                answer=original_answer,
                validated_sources=tuple(
                    validated_sources
                ),
            )
            

        grounded_answer_blocked = (
            not validated_sources
            and not self._is_abstention(
                original_answer
            )
        )

        final_sources = (
            []
            if grounded_answer_blocked
            else validated_sources
        )

        # --------------------------------------------------
        # 10. TRACE
        #
        # Trace the actual answer returned by the service.
        # This ensures evaluation measures production
        # behavior after the grounding gate.
        # --------------------------------------------------

        if self.trace_enabled:

            self.last_trace = RAGTrace(
                original_query=request.query,

                retrieval_query=retrieval_query,

                retrieved_results=tuple(retrieved),

                reranked_response=reranked,

                context=context,

                generated_answer=final_answer,

                validated_sources=tuple(final_sources),
            )

        # --------------------------------------------------
        # 11. STORE CURRENT TURN
        #
        # Store the original user query and the actual safe
        # answer returned by the service.
        # --------------------------------------------------

        self.memory_service.add_turn(
            tenant_id=request.user.tenant_id,
            conversation_id=request.conversation_id,
            user_message=request.query,
            assistant_message=final_answer,
        )
        # --------------------------------------------------
        # 12. RESPONSE METADATA
        # --------------------------------------------------

        retrieved_sources = list(
            dict.fromkeys(
                result.metadata.get(
                    "source_filename"
                )
                for result in retrieved
                if result.metadata.get(
                    "source_filename"
                )
            )
        )
        
        reranked_sources = list(
            dict.fromkeys(
                result
                .retrieval_result
                .metadata
                .get(
                    "source_filename"
                )
                for result
                in reranked.results
                if result
                .retrieval_result
                .metadata
                .get(
                    "source_filename"
                )
            )
        )

        context_sources = list(
            dict.fromkeys(
                chunk.source_filename
                for chunk in context.chunks
                if chunk.source_filename
            )
        )

        # --------------------------------------------------
        # 14. BUILD RESPONSE METADATA
        # --------------------------------------------------

        response_metadata = {

            "retrieved": len(retrieved),

            "reranked": len(reranked.results),

            "context_chunks": len(context.chunks),

            "context_tokens": context.total_tokens,

            "retrieved_sources": retrieved_sources,

            "reranked_sources": reranked_sources,

            "context_sources": context_sources,

            "retrieval_query": retrieval_query,

            "query_contextualized": contextualization.was_contextualized,

            "contextualizer_tokens": contextualization.total_tokens,

            "memory_turns_used": memory.total_turns,

            "answer_grounded": not grounded_answer_blocked,

            "grounded_answer_blocked": grounded_answer_blocked,

            "validated_citation_count": len(final_sources),

            "model": llm_response.model_name,

            "prompt_tokens": llm_response.prompt_tokens,

            "completion_tokens": llm_response.completion_tokens,

            "total_tokens": llm_response.total_tokens,

            "route": "rag",

            "authorized_departments": list(scope.departments),

            "effective_query": effective_query,

            "cache_hit": False,

            "cache_type": "semantic",
        }

        answer_departments = list(
            {
                str(
                    chunk.metadata.get(
                        "department"
                    )
                ).lower()
                for chunk in context.chunks
                if chunk.metadata.get(
                    "department"
                )
            }
        )
        # --------------------------------------------------
        # 16. SEMANTIC CACHE STORE
        # --------------------------------------------------

        if (
            not grounded_answer_blocked
            and final_sources
            and not self._is_abstention(
                final_answer
            )
        ):

            cache_id = str(uuid4())

            cache_saved = (
                self.cache_service.set_response(
                    cache_id=cache_id,
                    value={
                        "answer": (
                            final_answer
                        ),
                        "sources": (
                            final_sources
                        ),
                        "metadata": (
                            response_metadata
                        ),
                    },
                )
            )

            if cache_saved:

                self.semantic_cache_store.store(
                    vector=query_vector,
                    cache_id=cache_id,
                    tenant_id=str(request.user.tenant_id),
                    answer_departments=answer_departments,
                    query=effective_query,
                )
        # --------------------------------------------------
        # 17. STORE CONVERSATION MEMORY
        # --------------------------------------------------

        self.memory_service.add_turn(
            tenant_id=request.user.tenant_id,
            conversation_id=request.conversation_id,
            user_message=request.query,
            assistant_message=final_answer,
        )

        # --------------------------------------------------
        # 18. RESPONSE
        # --------------------------------------------------
        return RAGResponse(
            answer=final_answer,

            sources=final_sources,

            metadata = response_metadata,
        )

    def _is_abstention(
        self,
        answer: str,
    ) -> bool:

        return (
            answer.strip().lower()
            == (
                self.grounded_answer_gate
                .ABSTENTION_TEXT
                .lower()
            )
        )
    
    def _load_memory(
        self,
        request: RAGRequest,
    ):

        return (
            self.memory_service.get_memory(
                MemoryRequest(
                    tenant_id=request.user.tenant_id,
                    conversation_id=request.conversation_id,
                )
            )
        )
    
    def _contextualize(
        self,
        request: RAGRequest,
        memory,
    ):

        return (
            self.contextualization_service
            .contextualize(

                ContextualizationRequest(

                    query=request.query,

                    turns=memory.turns,
                )
            )
        )
    
    def _authorize(
        self,
        request: RAGRequest,
    ):

        if request.user.role in (
            "admin",
            "knowledge_admin",
        ):

            requested_departments = (
                "*",
            )

        else:

            requested_departments = request.user.department

        return (
            self.authorization_service
            .filter_departments(
                tenant_id=request.user.tenant_id,
                requested_departments=requested_departments,
                user_department=request.user.department,
                user_role=request.user.role,
            )
        )