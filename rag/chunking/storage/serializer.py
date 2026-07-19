import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from rag.chunking.schemas import Chunk


class ChunkSerializer:

    @staticmethod
    def chunk_to_dict(
        chunk: Chunk,
    ) -> dict:

        return asdict(chunk)

    @staticmethod
    def chunk_from_dict(
        data: dict,
    ) -> Chunk:

        return Chunk(**data)

    @classmethod
    def write_jsonl(
        cls,
        chunks,
        output_path: Path,
    ) -> None:

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_path = cls._create_temp_path(
            output_path
        )

        try:

            with temp_path.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as file:

                for chunk in chunks:

                    payload = cls.chunk_to_dict(
                        chunk
                    )

                    line = json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )

                    file.write(line)
                    file.write("\n")

                file.flush()

                os.fsync(file.fileno())

            os.replace(
                temp_path,
                output_path,
            )

        except Exception:

            if temp_path.exists():

                temp_path.unlink()

            raise

    @classmethod
    def load_jsonl(
        cls,
        input_path: Path,
    ) -> list[Chunk]:

        input_path = Path(input_path)

        chunks = []

        with input_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1,
            ):

                if not line.strip():
                    continue

                try:

                    data = json.loads(line)

                    chunk = cls.chunk_from_dict(
                        data
                    )

                except Exception as exc:

                    raise ValueError(
                        "Failed to deserialize chunk "
                        f"at line {line_number} in "
                        f"{input_path}: {exc}"
                    ) from exc

                chunks.append(chunk)

        return chunks

    @staticmethod
    def _create_temp_path(
        output_path: Path,
    ) -> Path:

        file_descriptor, temp_name = (
            tempfile.mkstemp(
                prefix=(
                    f".{output_path.name}."
                ),
                suffix=".tmp",
                dir=output_path.parent,
            )
        )

        os.close(file_descriptor)

        return Path(temp_name)