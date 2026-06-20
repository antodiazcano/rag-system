"""Script to create the final prompt to the LLM."""


def create_prompt(original_query: str, chunks: list[str]) -> str:
    """Creates the final prompt for the LLM including the retrieved chunks.

    Args:
        original_query: Original query of the user.
        chunks: Chunks to include (from most to least score).

    Returns:
        Prompt after RAG.
    """

    prompt = f"Original query of the user: {original_query}\n\nRetrieved chunks:"

    for chunk in chunks:
        prompt += f"\n\n{chunk}"

    return prompt
