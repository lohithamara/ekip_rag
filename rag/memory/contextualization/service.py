from rag.llm.schemas import (
    LLMRequest,
)

from rag.llm.service import (
    LLMService,
)

from rag.memory.contextualization.config import (
    ContextualizerConfig,
)

from rag.memory.contextualization.contextualizer import (
    ConversationQueryContextualizer,
)

from rag.memory.contextualization.schemas import (
    ContextualizationRequest,
    ContextualizationResponse,
)

from rag.memory.schemas import (
    ConversationTurn,
)


class ConversationContextualizationService:

    def __init__(
        self,
        llm_service: LLMService,
        config: ContextualizerConfig,
    ):
        self.llm_service = llm_service
        self.config = config

    def contextualize(
        self,
        request: ContextualizationRequest,
    ) -> ContextualizationResponse:

        query = request.query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        if not self.config.enabled:
            return self._fallback(query)

        if self.config.max_history_turns < 1:
            raise ValueError(
                "max_history_turns must be "
                "at least 1."
            )

        if (
            self.config.max_history_characters
            < 1
        ):
            raise ValueError(
                "max_history_characters must "
                "be at least 1."
            )

        if self.config.max_tokens < 1:
            raise ValueError(
                "max_tokens must be at least 1."
            )

        turns = self._select_history(
            request.turns
        )

        if not turns:
            return self._fallback(query)

        system_prompt, user_prompt = (
            ConversationQueryContextualizer
            .build_prompt(
                query=query,
                turns=turns,
            )
        )

        llm_response = (
            self.llm_service.generate(
                LLMRequest(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=(
                        self.config.temperature
                    ),
                    max_tokens=(
                        self.config.max_tokens
                    ),
                )
            )
        )

        contextualized_query = (
            self._clean_output(
                llm_response.answer
            )
        )

        if not self._is_valid_output(
            original_query=query,
            contextualized_query=(
                contextualized_query
            ),
        ):
            return self._fallback(
                query=query,
                model_name=(
                    llm_response.model_name
                ),
                prompt_tokens=(
                    llm_response.prompt_tokens
                ),
                completion_tokens=(
                    llm_response
                    .completion_tokens
                ),
                total_tokens=(
                    llm_response.total_tokens
                ),
            )

        return ContextualizationResponse(
            original_query=query,
            contextualized_query=(
                contextualized_query
            ),
            was_contextualized=(
                self._normalize(
                    contextualized_query
                )
                != self._normalize(query)
            ),
            model_name=(
                llm_response.model_name
            ),
            prompt_tokens=(
                llm_response.prompt_tokens
            ),
            completion_tokens=(
                llm_response.completion_tokens
            ),
            total_tokens=(
                llm_response.total_tokens
            ),
        )

    def _select_history(
        self,
        turns: tuple[
            ConversationTurn,
            ...,
        ],
    ) -> tuple[
        ConversationTurn,
        ...,
    ]:

        candidates = turns[
            -self.config.max_history_turns:
        ]

        selected = []

        total_characters = 0

        for turn in reversed(candidates):

            turn_characters = (
                len(turn.user_message)
                + len(turn.assistant_message)
            )

            if (
                total_characters
                + turn_characters
                > self.config
                .max_history_characters
            ):
                continue

            selected.append(turn)

            total_characters += (
                turn_characters
            )

        selected.reverse()

        return tuple(selected)

    @staticmethod
    def _clean_output(
        output: str,
    ) -> str:

        cleaned = output.strip()

        if (
            len(cleaned) >= 2
            and cleaned[0] == cleaned[-1]
            and cleaned[0] in {'"', "'"}
        ):
            cleaned = cleaned[1:-1].strip()

        prefixes = (
            "standalone query:",
            "standalone retrieval query:",
            "rewritten query:",
            "contextualized query:",
            "retrieval query:",
        )

        lowered = cleaned.lower()

        for prefix in prefixes:

            if lowered.startswith(prefix):

                cleaned = cleaned[
                    len(prefix):
                ].strip()

                break

        return " ".join(
            cleaned.split()
        )

    @staticmethod
    def _is_valid_output(
        original_query: str,
        contextualized_query: str,
    ) -> bool:

        if not contextualized_query:
            return False

        if len(contextualized_query) > 1000:
            return False

        if (
            "<conversation_history>"
            in contextualized_query
        ):
            return False

        if (
            "<current_query>"
            in contextualized_query
        ):
            return False

        if (
            "<task>"
            in contextualized_query
        ):
            return False

        if "\n" in contextualized_query:
            return False

        return True

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:

        return " ".join(
            text.lower().split()
        )

    @staticmethod
    def _fallback(
        query: str,
        model_name: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> ContextualizationResponse:

        return ContextualizationResponse(
            original_query=query,
            contextualized_query=query,
            was_contextualized=False,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=(
                completion_tokens
            ),
            total_tokens=total_tokens,
        )