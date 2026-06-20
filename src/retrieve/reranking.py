"""Script to rerank the results of the search."""

from abc import ABC, abstractmethod

from sentence_transformers import CrossEncoder


class Reranker(ABC):
    """Interface for a reranker."""

    def __init__(self, k: int) -> None:
        """Constructor of the class.

        Args:
            k: Number of chunks to return.
        """

        self.k = k

    @abstractmethod
    def rerank(self, query: str, chunks: list[str]) -> list[str]:
        """Reranks the chunks obtained.

        Args:
            query: Query used to obtain the chunks.
            chunks: Chunks of text to rerank.

        Returns:
            Reranked chunks.
        """


class BaselineReranker(Reranker):
    """Class for the baseline reranker, which basically does not do anything: it returns
    the chunks in the same order."""

    def rerank(self, query: str, chunks: list[str]) -> list[str]:
        """Reranks the chunks obtained.

        Args:
            query: Query used to obtain the chunks.
            chunks: Chunks of text to rerank.

        Returns:
            Reranked chunks.
        """

        return chunks[: self.k]


class CrossEncoderReranker(Reranker):
    """Class to implement the cross-encoder reranker, which uses a model that takes as
    input each chunk and the original query and gives a new score."""

    def __init__(
        self, k: int, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> None:
        """Constructor of the class.

        Args:
            k: Number of chunks to return.
            model_name: Name of the cross-encoder model to use.
        """

        super().__init__(k)

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[str]) -> list[str]:
        """Reranks the chunks obtained.

        Args:
            query: Query used to obtain the chunks.
            chunks: Chunks of text to rerank.

        Returns:
            Reranked chunks.
        """

        pairs = [[query, chunk] for chunk in chunks]
        scores = self.model.predict(pairs)

        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [chunk for chunk, _ in scored][: self.k]
