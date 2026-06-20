"""Script to define some utilities used."""

from pinecone import Index, Pinecone
from sentence_transformers import SentenceTransformer

from src.config import config


def embed_chunks(
    chunks: list[str], embedding_model: str = "all-MiniLM-L6-v2"
) -> list[list[float]]:
    """Embeds the chunks.

    Args:
        chunks: Chunks of text to embed.
        embedding_model: Model used to embed the chunks.

    Returns:
        Embedded chunks.
    """

    model = SentenceTransformer(embedding_model)
    return model.encode(chunks, normalize_embeddings=True).tolist()


def get_index_vector_db() -> Index:
    """Obtains the index of the db.

    Returns:
        Index of the db.

    Raises:
        ValueError: If the index name of the db is not found in the environment.
        TypeError: If the type of the index is not correct.
    """

    pc = Pinecone(api_key=config.vector_db.pinecone_api_key)

    if not isinstance(config.vector_db.pinecone_index_name, str):
        raise ValueError("Not index name found in the environment!")

    index = pc.Index(config.vector_db.pinecone_index_name)

    if not isinstance(index, Index):
        raise TypeError("Obtained GrpcIndex, not Index!")

    return index
