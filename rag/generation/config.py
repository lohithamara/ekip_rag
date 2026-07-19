from dataclasses import dataclass


@dataclass(slots=True)
class ContextBuilderConfig:

    # Maximum tokens sent to the LLM
    max_context_tokens: int = 3000

    # Include neighbouring chunks
    include_adjacent_chunks: bool = True

    # Number of chunks before/after
    adjacent_chunk_window: int = 1

    # Remove duplicate chunks
    deduplicate: bool = True

    # Preserve document order after assembly
    preserve_document_order: bool = True