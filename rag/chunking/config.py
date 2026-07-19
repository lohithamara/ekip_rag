from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChunkingConfig:

    # ----------------------------------
    # Strategy
    # ----------------------------------

    strategy: str

    chunking_version: str = "1.0"

    # ----------------------------------
    # Chunk size
    # ----------------------------------

    chunk_size: int = 512

    chunk_overlap: int = 64

    min_chunk_size: int = 50

    # ----------------------------------
    # Size measurement
    # ----------------------------------

    length_unit: str = "tokens"

    tokenizer_name: str | None = None

    # ----------------------------------
    # Recursive splitting
    # ----------------------------------

    separators: tuple[str, ...] = (
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    )

    # ----------------------------------
    # Provenance behavior
    # ----------------------------------

    preserve_page_boundaries: bool = True

    preserve_section_boundaries: bool = True

    preserve_table_boundaries: bool = True

    # ----------------------------------
    # Small chunk handling
    # ----------------------------------

    merge_small_chunks: bool = True

    # ----------------------------------
    # Table behavior
    # ----------------------------------

    table_chunking_mode: str = "row_group"

    table_rows_per_chunk: int = 25

    repeat_table_headers: bool = True

    # ----------------------------------
    # Strategy-specific configuration
    # ----------------------------------

    extra: dict = field(
        default_factory=dict
    )

    def __post_init__(self):

        if self.chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than 0."
            )

        if self.chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller "
                "than chunk_size."
            )

        if self.min_chunk_size < 0:
            raise ValueError(
                "min_chunk_size cannot be negative."
            )

        if self.length_unit not in {
            "characters",
            "tokens",
        }:
            raise ValueError(
                "length_unit must be "
                "'characters' or 'tokens'."
            )

        if self.table_chunking_mode not in {
            "whole_table",
            "row_group",
        }:
            raise ValueError(
                "Unsupported table_chunking_mode."
            )

        if self.table_rows_per_chunk <= 0:
            raise ValueError(
                "table_rows_per_chunk must "
                "be greater than 0."
            )