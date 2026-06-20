"""Script to build the complete RAG pipeline: transformation + search + reranking."""

from src.retrieve.context import create_prompt
from src.retrieve.reranking import CrossEncoderReranker, Reranker
from src.retrieve.search import HybridSearch, Search
from src.retrieve.transformation import ExpansionQueryTransformer, QueryTransformer
from src.utils import get_index_vector_db


def rag_pipeline(
    query: str,
    *,
    query_transformer: QueryTransformer,
    searcher: Search,
    reranker: Reranker,
) -> str:
    """Executes all the RAG pipeline to obtain the final prompt.

    Args:
        query: Original query of the user.
        query_transformer: Query transformer.
        searcher: Search method used in the vector db.
        reranker: Reranker of the scores of the chunks.

    Returns:
        Final prompt using RAG.
    """

    transformed_query = query_transformer.transform(query)
    chunks = searcher.search(transformed_query)
    reranked_chunks = reranker.rerank(transformed_query, chunks)
    final_prompt = create_prompt(query, reranked_chunks)

    return final_prompt


def main(query: str) -> str:
    """Main function.

    Args:
        query: Query of the user.

    Returns:
        Chunks obtained.
    """

    pinecone_index = get_index_vector_db()
    k = 3

    query_transformer = ExpansionQueryTransformer()
    searcher = HybridSearch(pinecone_index, 2 * k)
    reranker = CrossEncoderReranker(k)

    return rag_pipeline(
        query, query_transformer=query_transformer, searcher=searcher, reranker=reranker
    )
