import json
from dataclasses import asdict
from pathlib import Path

from rag.embeddings.schemas import EmbeddingRecord


class EmbeddingSerializer:

    @staticmethod
    def write_jsonl(
        records: tuple[EmbeddingRecord, ...],
        file_path: Path,
    ) -> None:

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with file_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            for record in records:

                file.write(
                    json.dumps(
                        asdict(record),
                        ensure_ascii=False,
                    )
                )

                file.write("\n")

    @staticmethod
    def load_jsonl(
        file_path: Path,
    ) -> tuple[EmbeddingRecord, ...]:

        records = []

        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                data = json.loads(line)

                data["vector"] = tuple(
                    data["vector"]
                )

                records.append(
                    EmbeddingRecord(**data)
                )

        return tuple(records)