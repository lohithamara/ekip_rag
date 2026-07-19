from ingestion.schemas.documents import CleanDocument

from rag.chunking.base import BaseChunkingStrategy
from rag.chunking.chunk_factory import ChunkFactory
from rag.chunking.config import ChunkingConfig
from rag.chunking.schemas import ChunkingResult
from rag.chunking.tokenization.base import BaseTokenCounter


class RecursiveChunkingStrategy(BaseChunkingStrategy):

    def __init__(
        self,
        config: ChunkingConfig,
        token_counter: BaseTokenCounter,
        chunk_factory: ChunkFactory,
    ):
        if config.strategy != self.name:
            raise ValueError(
                f"Expected strategy='{self.name}', "
                f"received '{config.strategy}'."
            )

        self.config = config
        self.token_counter = token_counter
        self.chunk_factory = chunk_factory
    STRATEGY_NAME = "recursive"

    @property
    def name(self) -> str:
        return self.STRATEGY_NAME

    def chunk(
        self,
        document: CleanDocument,
    ) -> ChunkingResult:

        text = document.clean_text

        if not text or not text.strip():
            return self._build_result(document, [])

        pieces = self._recursive_split(
            text=text,
            separators=self.config.separators,
        )

        pieces = self._merge_pieces(pieces)

        chunks = []

        search_start = 0

        for chunk_index, chunk_text in enumerate(pieces):

            start_char = text.find(
                chunk_text,
                search_start,
            )

            if start_char == -1:
                start_char = None
                end_char = None

            else:
                end_char = (
                    start_char + len(chunk_text)
                )

                search_start = start_char

            chunks.append(
                self.chunk_factory.create_chunk(
                    document=document,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    metadata={
                        "length_unit": (
                            self.config.length_unit
                        ),
                        "separators": list(
                            self.config.separators
                        ),
                    },
                )
            )

        return self._build_result(
            document=document,
            chunks=chunks,
        )

    def _recursive_split(
        self,
        text: str,
        separators: tuple[str, ...],
    ) -> list[str]:

        if (
            self.token_counter.count(text)
            <= self.config.chunk_size
        ):
            return [text]

        if not separators:
            return self._hard_split(text)

        separator = separators[0]

        if separator == "":
            return self._hard_split(text)

        parts = text.split(separator)

        if len(parts) == 1:
            return self._recursive_split(
                text=text,
                separators=separators[1:],
            )

        pieces = []

        for index, part in enumerate(parts):

            if not part:
                continue

            if index < len(parts) - 1:
                candidate = part + separator
            else:
                candidate = part

            if (
                self.token_counter.count(candidate)
                <= self.config.chunk_size
            ):
                pieces.append(candidate)

            else:
                pieces.extend(
                    self._recursive_split(
                        text=candidate,
                        separators=separators[1:],
                    )
                )

        return pieces

    def _hard_split(
        self,
        text: str,
    ) -> list[str]:

        if self.config.length_unit == "characters":

            return [
                text[start:start + self.config.chunk_size]
                for start in range(
                    0,
                    len(text),
                    self.config.chunk_size,
                )
            ]

        words = text.split()

        return [
            " ".join(
                words[
                    start:
                    start + self.config.chunk_size
                ]
            )
            for start in range(
                0,
                len(words),
                self.config.chunk_size,
            )
        ]

    def _merge_pieces(
        self,
        pieces: list[str],
    ) -> list[str]:

        merged = []

        current_text = ""

        for piece in pieces:

            if not piece or not piece.strip():
                continue

        # Defensive guarantee:
        # recursive splitting should already ensure
        # this, but never trust intermediate output.
            if (
                self.token_counter.count(piece)
                > self.config.chunk_size
            ):
                subpieces = self._hard_split(piece)

                if subpieces == [piece]:
                    raise RuntimeError(
                        "Unable to split oversized "
                        "recursive piece."
                    )   

                # Process safely through the same
                # merge logic.
                remaining_pieces = (
                    subpieces
                )

            else:
                remaining_pieces = [piece]

            for safe_piece in remaining_pieces:

                if not current_text:

                    current_text = safe_piece

                    continue

                candidate = (
                    current_text + safe_piece
                )

                if (
                    self.token_counter.count(candidate)
                    <= self.config.chunk_size
                ):

                    current_text = candidate

                    continue

                # Current chunk is complete.
                merged.append(
                    current_text.strip()
                )

                overlap_text = self._build_overlap(
                    current_text
                )

                # The requested overlap may be too large
                # for the next piece.
                #
                # Trim overlap until:
                #
                # overlap + safe_piece <= chunk_size

                overlap_text = (
                    self._fit_overlap_to_piece(
                        overlap_text=overlap_text,
                        piece=safe_piece,
                    )
                )

                current_text = (
                    overlap_text + safe_piece
                )

                # Final defensive invariant.
                if (
                    self.token_counter.count(
                        current_text
                    )
                    > self.config.chunk_size
                ):
                    raise RuntimeError(
                        "Recursive chunk exceeded "
                        "chunk_size after overlap fitting."
                    )

        if current_text and current_text.strip():

            merged.append(
                current_text.strip()
            )

        return self._handle_small_tail(merged)

    def _build_overlap(
        self,
        text: str,
    ) -> str:

        overlap = self.config.chunk_overlap

        if overlap <= 0:
            return ""

        if self.config.length_unit == "characters":
            return text[-overlap:]

        words = text.split()

        return " ".join(
            words[-overlap:]
        )

    def _handle_small_tail(
        self,
        chunks: list[str],
    ) -> list[str]:

        if len(chunks) < 2:
            return chunks

        last_length = self.token_counter.count(
            chunks[-1]
        )

        if (
            last_length >= self.config.min_chunk_size
            or not self.config.merge_small_chunks
        ):
            return chunks

        previous = chunks[-2]
        tail = chunks[-1]

        combined = f"{previous}\n{tail}".strip()

        # Do not create an oversized chunk merely
        # to eliminate a small tail.
        if (
            self.token_counter.count(combined)
            <= self.config.chunk_size
        ):
            chunks[-2] = combined
            chunks.pop()

        # Otherwise retain the small tail so unique
        # source content is never silently discarded.

        return chunks

    def _build_result(
        self,
        document: CleanDocument,
        chunks: list,
    ) -> ChunkingResult:

        return ChunkingResult(
            document_id=document.metadata.document_id,
            tenant_id=document.metadata.tenant_id,
            strategy=self.name,
            chunking_version=(
                self.config.chunking_version
            ),
            chunks=chunks,
            total_chunks=len(chunks),
            total_characters=sum(
                chunk.character_count
                for chunk in chunks
            ),
            total_tokens=sum(
                chunk.token_count
                for chunk in chunks
            ),
            metadata={
                "length_unit": self.config.length_unit,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": (
                    self.config.chunk_overlap
                ),
                "min_chunk_size": (
                    self.config.min_chunk_size
                ),
                "merge_small_chunks": (
                    self.config.merge_small_chunks
                ),
                "token_counter": (
                    self.token_counter.name
                ),
                "separators": list(
                    self.config.separators
                ),
            },
        )
    def _fit_overlap_to_piece(
        self,
        overlap_text: str,
        piece: str,
    ) -> str:

        if not overlap_text:
            return ""

        available_size = (
            self.config.chunk_size
            - self.token_counter.count(piece)
        )

        if available_size <= 0:
            return ""

        if self.config.length_unit == "characters":

            # Use the available character budget.
            return overlap_text[
                -available_size:
            ]

        words = overlap_text.split()

        if not words:
            return ""

        # Keep the largest suffix that fits.
        low = 0
        high = min(
            len(words),
            available_size,
        )

        best = ""

        while low <= high:

            middle = (
                low + high
            ) // 2

            candidate = " ".join(
                words[-middle:]
            ) if middle else ""

            combined = candidate + piece

            if (
                self.token_counter.count(combined)
                <= self.config.chunk_size
            ):

                best = candidate

                low = middle + 1

            else:

                high = middle - 1

        if best:
            best += " "

        return best