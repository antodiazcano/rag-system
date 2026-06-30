"""Script to define some utilities used."""

from pinecone import Index, Pinecone

from src.config import config


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
