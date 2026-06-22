"""Script so save the embeddings of the chunks of the documents in a vector db."""

import uuid

from pinecone import Index

from src.chunk.chunks import DocumentProcessor, FixedWindowChunker
from src.utils import get_index_vector_db


def save_to_db(
    chunks: list[str],
    embeddings: list[list[float]],
    pinecone_index: Index,
    batch_size: int = 50,
) -> None:
    """Saves chunks and their embeddings into the vector db.

    Args:
        chunks: List of chunk texts.
        embeddings: List of embedding vectors, one per chunk (same order).
        pinecone_index: Pinecone index to upsert into.
        batch_size: Number of vectors per upsert call.
    """

    vectors = [
        {"id": str(uuid.uuid4()), "values": emb, "metadata": {"text": chunk}}
        for chunk, emb in zip(chunks, embeddings)
    ]

    for i in range(0, len(vectors), batch_size):
        pinecone_index.upsert(vectors=vectors[i : i + batch_size])


def main(delete_all: bool = True) -> None:
    """Main function.

    Args:
        delete_all: True to delete the previous embeddings of the db, False otherwise.
    """

    path = "docs/coffee.pdf"
    processor = DocumentProcessor(path, chunker=FixedWindowChunker())
    chunks, embeddings = processor.process()
    index = get_index_vector_db()
    if delete_all:
        index.delete(delete_all=True)
    save_to_db(chunks, embeddings, pinecone_index=index)


if __name__ == "__main__":
    main()
