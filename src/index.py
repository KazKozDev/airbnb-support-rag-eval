"""Build the Chroma vector index from data/chunks.jsonl."""
import json

import chromadb
from chromadb.utils import embedding_functions

from src import config


def load_chunks() -> list[dict]:
    with open(config.CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def get_collection():
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL
    )
    return client.get_or_create_collection("docqa", embedding_function=ef)


def main():
    chunks = load_chunks()
    col = get_collection()
    # rebuild from scratch so re-chunking doesn't leave stale ids
    existing = col.get()["ids"]
    if existing:
        col.delete(ids=existing)
    BATCH = 64
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        col.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{"page": c["page"]} for c in batch],
        )
    print(f"Indexed {len(chunks)} chunks into Chroma at {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
