"""Script to build the complete RAG pipeline: transformation + search + reranking."""

from src.retrieve.context import create_prompt
from src.retrieve.reranking import Reranker
from src.retrieve.search import Search
from src.retrieve.transformation import QueryTransformer


class RAGPipeline:
    """Class to execute the RAG pipeline."""

    def __init__(
        self, query_transformer: QueryTransformer, searcher: Search, reranker: Reranker
    ) -> None:
        """Constructor of the class.

        Args:
            query_transformer: Query transformer.
            searcher: Search method used in the vector db.
            reranker: Reranker of the scores of the chunks.
        """

        self.query_transformer = query_transformer
        self.searcher = searcher
        self.reranker = reranker

    def execute_pipeline(self, query: str) -> tuple[str, list[str]]:
        """Executes all the RAG pipeline to obtain the final prompt.

        Args:
            query: Original query of the user.

        Returns:
            Tuple of (final prompt using RAG, reranked chunks).
        """

        transformed_query = self.query_transformer.transform(query)
        chunks = self.searcher.search(transformed_query)
        reranked_chunks = self.reranker.rerank(transformed_query, chunks)
        final_prompt = create_prompt(query, reranked_chunks)

        return final_prompt, reranked_chunks
