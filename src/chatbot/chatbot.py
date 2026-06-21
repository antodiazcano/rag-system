"""Script to create the chatbot."""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.retrieve.pipeline import RAGPipeline


class Chatbot:
    """Class that defines the chatbot to interact with the user."""

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7,
    ) -> None:
        """Constructor of the class.

        Args:
            rag_pipeline: RAG pipeline used.
            model: Name of the model used.
            temperature: Temperature of the model.
        """

        self.llm = ChatGroq(model=model, temperature=temperature)
        self.history: list[BaseMessage] = [
            SystemMessage(content=self._get_system_prompt())
        ]
        self.rag_pipeline = rag_pipeline
        self.last_chunks: list[str] = []

    @staticmethod
    def _get_system_prompt() -> str:
        """Creates the system prompt.

        Returns:
            System prompt.
        """

        return (
            "You are an expert, fact-driven AI Assistant. Your task is to provide "
            "accurate, concise, and structured answers to the User Query based solely "
            "on the provided Grounding Context.\n\n"
            "### Core Instructions:"
            "1. **Strict Contextual Grounding:** Rely ONLY on the clear facts directly "
            "mentioned in the Grounding Context. Do not assume, extrapolate, or bring "
            "in outside knowledge not present in the text.\n"
            "2. **Handle Incomplete Information:** If the context does not contain the "
            "answer, or if the context is insufficient to form a complete response, "
            "state clearly and concisely: 'I am sorry, but the provided documentation "
            "does not contain enough information to answer this question.' Do not "
            "attempt to guess or synthesize an answer from general knowledge.\n"
            "3. **In-Line Citations:** You must back up your statements using in-line "
            "citations referencing the Source ID from the context (e.g., '[Source 1]', "
            "'[Source 2]'). Place the citation at the end of the sentence or clause it "
            "validates.\n"
            "4. **Formatting:** Use clear markdown formatting (bullet points, bold "
            "text, or code blocks) to make the information scannable and easy to read. "
            "Keep the tone professional, objective, and neutral."
        )

    def chat(self, query: str) -> str:
        """Obtains a response of the model for a query.

        Args:
            query: Query of the user.

        Returns:
            Answer of the model.
        """

        rag_prompt, chunks = self.rag_pipeline.execute_pipeline(query)
        # rag_prompt, chunks = query, []
        self.last_chunks = chunks
        self.history.append(HumanMessage(content=rag_prompt))
        response = self.llm.invoke(self.history)
        self.history.append(AIMessage(content=response.content))

        content = response.content
        if not isinstance(content, str):
            raise RuntimeError("Error executing the LLM!")

        return content
