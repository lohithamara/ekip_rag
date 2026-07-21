import re

from rag.generation.config import (
    ContextBuilderConfig,
)
from rag.generation.schemas import (
    ContextChunk,
    ContextRequest,
    ContextResponse,
)


class ContextBuilder:

    def __init__(
        self,
        config: ContextBuilderConfig,
    ):
        self.config = config

    def build(
        self,
        request: ContextRequest,
    ) -> ContextResponse:

        chunks = self._convert_chunks(
            request
        )

        if self.config.deduplicate:
            chunks = self._deduplicate(
                chunks
            )

        chunks = self._apply_budget(
            chunks
        )

        return ContextResponse(
            chunks=chunks,
            total_chunks=len(chunks),
            total_tokens=sum(
                chunk.token_count
                for chunk in chunks
            ),
        )

    def _convert_chunks(
        self,
        request: ContextRequest,
    ) -> list[ContextChunk]:

        converted = []

        for result in request.reranked_results:

            retrieval = (
                result.retrieval_result
            )

            converted.append(
                ContextChunk(
                    chunk_id=retrieval.chunk_id,
                    document_id=retrieval.document_id,
                    source_filename=(
                        retrieval.metadata.get(
                            "source_filename",
                            "",
                        )
                    ),
                    text=retrieval.text,
                    token_count=retrieval.metadata.get(
                        "token_count",
                        0,
                    ),
                    metadata=retrieval.metadata,
                )
            )

        return converted

    def _deduplicate(
        self,
        chunks: list[ContextChunk],
    ) -> list[ContextChunk]:

        seen_chunk_ids = set()
        seen_content = set()

        unique = []

        for chunk in chunks:

            if chunk.chunk_id in seen_chunk_ids:
                continue

            normalized_text = " ".join(
                chunk.text.lower().split()
            )

            if normalized_text in seen_content:
                continue

            seen_chunk_ids.add(
                chunk.chunk_id
            )

            seen_content.add(
                normalized_text
            )

            unique.append(chunk)

        return unique

    def _is_complete_unit(self, text: str) -> bool:
        return bool(text.strip())

    def _select_relevant_content(
        self,
        query: str,
        chunks: list[ContextChunk],
    ) -> list[ContextChunk]:

        query_terms = self._extract_terms(
            query
        )

        if not query_terms:
            return chunks

        selected = []

        for chunk in chunks:

            units = self._split_content(
                chunk.text
            )

            scored_units = []

            for position, unit in enumerate(
                units
            ):

                if not self._is_complete_unit(unit):
                    continue

                unit_terms = self._extract_terms(
                    unit
                )

                overlap = len(
                    query_terms & unit_terms
                )

                if overlap <= 0:
                    continue

                scored_units.append(
                    (
                        overlap,
                        position,
                        unit,
                    )
                )

            if not scored_units:

                complete_units = [
                    unit
                    for unit in units
                    if self._is_complete_unit(unit)
                ]

                if not complete_units:
                    continue

                fallback_text = "\n\n".join(
                    complete_units[:3]
                )

                selected.append(
                    ContextChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        source_filename=(
                            chunk.source_filename
                        ),
                        text=fallback_text,
                        token_count=self._count_tokens(
                            fallback_text
                        ),
                        metadata=chunk.metadata,
                    )
                )

                continue

            scored_units.sort(
                key=lambda item: (
                    -item[0],
                    item[1],
                )
            )

            kept_units = scored_units[:3]

            kept_units.sort(
                key=lambda item: item[1]
            )

            selected_text = "\n\n".join(
                item[2]
                for item in kept_units
            )

            if not selected_text.strip():
                selected.append(chunk)
                continue

            selected.append(
                ContextChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source_filename=(
                        chunk.source_filename
                    ),
                    text=selected_text,
                    token_count=self._count_tokens(
                        selected_text
                    ),
                    metadata=chunk.metadata,
                )
            )

        return selected

    def _split_content(
        self,
        text: str,
    ) -> list[str]:

        blocks = re.split(
            r"\n\s*\n",
            text.strip(),
        )

        units = []

        for block in blocks:

            block = block.strip()

            if not block:
                continue

            lines = [
                line.strip()
                for line in block.splitlines()
                if line.strip()
            ]

            if self._looks_like_table(lines):

                units.extend(lines)

            else:

                units.append(block)

        return units

    def _looks_like_table(
        self,
        lines: list[str],
    ) -> bool:

        if len(lines) < 2:
            return False

        pipe_lines = sum(
            "|" in line
            for line in lines
        )

        return pipe_lines >= 2

    def _extract_terms(
        self,
        text: str,
    ) -> set[str]:

        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "how",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "that",
            "the",
            "to",
            "what",
            "when",
            "where",
            "which",
            "who",
            "with",
        }

        words = re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )

        return {
            word
            for word in words
            if (
                word not in stopwords
                and len(word) > 1
            )
        }

    def _count_tokens(
        self,
        text: str,
    ) -> int:

        return len(
            text.split()
        )

    def _apply_budget(
        self,
        chunks: list[ContextChunk],
    ) -> list[ContextChunk]:

        selected = []

        total_tokens = 0

        for chunk in chunks:

            if chunk.token_count <= 0:
                continue

            if (
                total_tokens
                + chunk.token_count
                > self.config.max_context_tokens
            ):
                continue

            selected.append(chunk)

            total_tokens += (
                chunk.token_count
            )

        return selected