"""Script to search the most similar chunks to the query in the vector db."""

import re
from abc import ABC, abstractmethod

from pinecone import Index
from rank_bm25 import BM25Okapi

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
            if (not isinstance(match.metadata, dict)) or ("text" not in match.metadata):
                continue
            matches.append(match.metadata["text"])

        return matches


class HybridSearch(Search):
    """Hybrid search combining BM25 keyword retrieval with vector search.

    Runs BM25 over a local corpus and vector search over Pinecone, then fuses
    the ranked lists using Reciprocal Rank Fusion (RRF). The corpus is
    automatically fetched from the Pinecone index at init time.
    """

    def __init__(self, pinecone_index: Index, k: int) -> None:
        """Constructor of the class.

        Args:
            pinecone_index: Pinecone index used for retrieval.
            k: Number of chunks to retrieve.
        """

        super().__init__(pinecone_index, k)

        self.corpus = self._fetch_corpus()
        self.bm25 = BM25Okapi([self._tokenize(doc) for doc in self.corpus])

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenizes a text into lowercased words.

        Args:
            text: Text to tokenize.

        Returns:
            Tokenized text.
        """

        return re.findall(r"\w+", text.lower())

    def _fetch_corpus(self) -> list[str]:
        """Fetches all chunk texts from the Pinecone index.

        Returns:
            List of all chunk texts stored in the index.
        """

        # Obtain all ids of the db
        ids: list[str] = []
        for page in self.pinecone_index.list():
            for item in page.vectors:
                if item.id is not None:
                    ids.append(item.id)

        # Add all chunks to the corpus
        texts: list[str] = []
        batch_size = 1_000
        for i in range(0, len(ids), batch_size):
            batch = ids[i : i + batch_size]
            result = self.pinecone_index.fetch(ids=batch)
            for vec in result.vectors.values():
                if vec.metadata is None:
                    continue
                texts.append(vec.metadata["text"])

        return texts

    def rrf(
        self, bm25_texts: list[str], vector_texts: list[str], k_constant: float = 60.0
    ) -> list[str]:
        """Applies RRF algorithm to obtain the best k chunks.

        Args:
            bm25 texts: Chunks obtained with BM25.
            vector_texts: Chunks obtained with vector search.
            k_constant: K constant of the algorithm.

        Returns:
            Top chunks using RRF algorithm.
        """

        rrf_scores: dict[str, float] = {}

        for rank, text in enumerate(bm25_texts[: self.k * 2]):
            rrf_scores[text] = rrf_scores.get(text, 0) + 1.0 / (k_constant + rank)

        for rank, text in enumerate(vector_texts):
            rrf_scores[text] = rrf_scores.get(text, 0) + 1.0 / (k_constant + rank)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        return [text for text, _ in ranked[: self.k]]

    def search(self, query: str) -> list[str]:
        """Searches the most similar chunks to the query using hybrid search.

        Args:
            query: Query of the user (which was modified in the 'transformation' step.

        Returns:
            Most relevant chunks for the query.
        """

        # BM25 results
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
        bm25_texts = [self.corpus[idx] for idx, _ in bm25_ranked]

        # Vector search results
        vector_texts = VectorSearch(self.pinecone_index, self.k * 2).search(query)

        # RRF algorithm
        return self.rrf(bm25_texts, vector_texts)
