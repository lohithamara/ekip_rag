import json
from pathlib import Path

from rag.chunking.storage.serializer import ChunkSerializer


EVALUATION_FILE = Path(
    "data/evaluation/retrieval_queries.json"
)

CHUNK_ROOT = Path("data/processed/chunks")

QUERY_NUMBERS = {25, 39}


def main():

    queries = json.loads(
        EVALUATION_FILE.read_text(
            encoding="utf-8"
        )
    )

    chunks = []

    for path in CHUNK_ROOT.rglob("chunks.jsonl"):
        chunks.extend(
            ChunkSerializer.load_jsonl(path)
        )

    for query_number in sorted(QUERY_NUMBERS):

        item = queries[query_number - 1]

        relevant = set(
            item["relevant_documents"]
        )

        matching_chunks = [
            chunk
            for chunk in chunks
            if chunk.source_filename in relevant
        ]

        print()
        print("=" * 70)
        print(f"QUERY {query_number}")
        print("=" * 70)
        print(f"Query      : {item['query']}")
        print(f"Department : {item['department']}")
        print(f"Relevant   : {item['relevant_documents']}")
        print(f"Chunks     : {len(matching_chunks)}")

        for index, chunk in enumerate(
            matching_chunks,
            start=1,
        ):
            print()
            print("-" * 70)
            print(f"CHUNK {index}")
            print(f"Chunk ID   : {chunk.chunk_id}")
            print(f"Chunk type : {chunk.chunk_type}")
            print(f"Strategy   : {chunk.strategy}")
            print(f"Tokens     : {chunk.token_count}")
            print()
            print(chunk.text)


if __name__ == "__main__":
    main()