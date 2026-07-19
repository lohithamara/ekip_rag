from dataclasses import dataclass, field
from typing import Any

from rag.chunking.routing.schemas import DocumentProfile


@dataclass(frozen=True)
class RoutingDecision:

    strategy: str

    content_scope: str

    reason: str

    priority: int

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class RoutingResult:

    document_id: str

    decisions: tuple[RoutingDecision, ...]

    router_version: str


class ChunkingRouter:

    ROUTER_VERSION = "1.0"

    TABULAR_FILE_TYPES = {
        ".csv",
        ".xlsx",
    }

    STRUCTURED_FILE_TYPES = {
        ".md",
        ".html",
        ".docx",
    }

    GENERIC_FILE_TYPES = {
        ".txt",
    }

    def route(
        self,
        profile: DocumentProfile,
    ) -> RoutingResult:

        decisions = []

        file_type = profile.file_type.lower()
        # ----------------------------------
        # JSON DOCUMENTS
        # ----------------------------------

        if file_type == ".json":

            decisions.append(
                RoutingDecision(
                    strategy="json_aware",
                    content_scope="text",
                    reason=(
                        "JSON document uses JSON-aware "
                        "chunking."
                    ),
                    priority=100,
                )
            )

            return self._build_result(
                profile,
                decisions,
            )
        # ----------------------------------
        # TABULAR DOCUMENTS
        # ----------------------------------

        if file_type in self.TABULAR_FILE_TYPES:

            if profile.has_tables:

                decisions.append(
                    RoutingDecision(
                        strategy="table_aware",
                        content_scope="tables",
                        reason=(
                            "Tabular file with extracted "
                            "table content."
                        ),
                        priority=100,
                    )
                )

            else:

                decisions.append(
                    RoutingDecision(
                        strategy="recursive",
                        content_scope="text",
                        reason=(
                            "Tabular file contains no "
                            "usable extracted tables."
                        ),
                        priority=50,
                    )
                )

            return self._build_result(
                profile,
                decisions,
            )

        # ----------------------------------
        # PDF DOCUMENTS
        # ----------------------------------

        if file_type == ".pdf":

            if profile.has_pages:

                decisions.append(
                    RoutingDecision(
                        strategy="page_aware",
                        content_scope="text",
                        reason=(
                            "PDF contains extracted "
                            "page-level text."
                        ),
                        priority=100,
                    )
                )

            elif profile.token_count > 0:

                decisions.append(
                    RoutingDecision(
                        strategy="recursive",
                        content_scope="text",
                        reason=(
                            "PDF has text but no usable "
                            "page-level representation."
                        ),
                        priority=50,
                    )
                )

            if profile.has_tables:

                decisions.append(
                    RoutingDecision(
                        strategy="table_aware",
                        content_scope="tables",
                        reason=(
                            "PDF contains extracted "
                            "tables requiring independent "
                            "table chunks."
                        ),
                        priority=90,
                    )
                )

            return self._build_result(
                profile,
                decisions,
            )

        # ----------------------------------
        # STRUCTURED DOCUMENTS
        # ----------------------------------

        if file_type in self.STRUCTURED_FILE_TYPES:

            if profile.has_structure:

                decisions.append(
                    RoutingDecision(
                        strategy="structure_aware",
                        content_scope="text",
                        reason=(
                            "Document contains sufficient "
                            "detected structural signals."
                        ),
                        priority=100,
                        metadata={
                            "structure_confidence": (
                                profile.structure_confidence
                            )
                        },
                    )
                )

            else:

                decisions.append(
                    RoutingDecision(
                        strategy="recursive",
                        content_scope="text",
                        reason=(
                            "Document structure confidence "
                            "is below routing threshold."
                        ),
                        priority=50,
                    )
                )

            if profile.has_tables:

                decisions.append(
                    RoutingDecision(
                        strategy="table_aware",
                        content_scope="tables",
                        reason=(
                            "Structured document also "
                            "contains extracted tables."
                        ),
                        priority=90,
                    )
                )

            return self._build_result(
                profile,
                decisions,
            )

        # ----------------------------------
        # GENERIC DOCUMENTS
        # ----------------------------------

        if file_type in self.GENERIC_FILE_TYPES:

            decisions.append(
                RoutingDecision(
                    strategy="recursive",
                    content_scope="text",
                    reason=(
                        "Generic text representation uses "
                        "recursive chunking fallback."
                    ),
                    priority=50,
                )
            )

            return self._build_result(
                profile,
                decisions,
            )

        # ----------------------------------
        # UNKNOWN FILE TYPE
        # ----------------------------------

        decisions.append(
            RoutingDecision(
                strategy="recursive",
                content_scope="text",
                reason=(
                    "Unknown file type uses safe "
                    "recursive fallback."
                ),
                priority=10,
            )
        )

        return self._build_result(
            profile,
            decisions,
        )

    def _build_result(
        self,
        profile: DocumentProfile,
        decisions: list[RoutingDecision],
    ) -> RoutingResult:

        if not decisions:

            raise ValueError(
                "ChunkingRouter produced no routing "
                f"decision for document "
                f"{profile.document_id}."
            )

        # Highest-priority decisions first.

        decisions.sort(
            key=lambda decision: (
                -decision.priority,
                decision.strategy,
            )
        )

        # Defensive duplicate protection.

        seen = set()

        unique_decisions = []

        for decision in decisions:

            identity = (
                decision.strategy,
                decision.content_scope,
            )

            if identity in seen:
                continue

            seen.add(identity)

            unique_decisions.append(decision)

        return RoutingResult(
            document_id=profile.document_id,
            decisions=tuple(unique_decisions),
            router_version=self.ROUTER_VERSION,
        )