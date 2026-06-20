"""Script to search the most similar chunks to the query in the vector db."""

from abc import ABC, abstractmethod

from pinecone import Index

from src.utils import embed_chunks


class Search(ABC):
    """base class for the search of the most similar chunks to the query."""

    def __init__(self, pinecone_index: Index, k: int) -> None:
        """Constructor of the class.

        Args:
            pinecone_index: Pinecone index used for retrieval.
            k: Number of chunks to retrieve.
        """

        self.pinecone_index = pinecone_index
        self.k = k

    @abstractmethod
    def search(self, query: str) -> list[str]:
        """Searches the most similar chunks to the query in the vector db.

        Args:
            query: Query of the user (which was modified in the 'transformation' step.

        Returns:
            Most relevant chunks for the query.
        """


class VectorSearch(Search):
    """Vector search of the most similar chunks to the query."""

    def search(self, query: str) -> list[str]:
        """Searches the most similar chunks to the query in the vector db.

        Args:
            query: Query of the user (which was modified in the 'transformation' step.

        Returns:
            Most relevant chunks for the query.
        """

        embedded_query = embed_chunks([query])[0]
        response = self.pinecone_index.query(
            vector=embedded_query, top_k=self.k, include_metadata=True
        )

        matches = []
        for match in response.matches:
            if (isinstance(match.metadata, dict)) and (
                "text" not in match.metadata.keys()
            ):
                continue
            matches.append(match.metadata["text"])

        return matches


class HybridSearch(Search):
    """Vector search of the most similar chunks to the query."""

    def search(self, query: str) -> list[str]:
        """Searches the most similar chunks to the query in the vector db.

        Args:
            query: Query of the user (which was modified in the 'transformation' step.

        Returns:
            Most relevant chunks for the query.
        """
