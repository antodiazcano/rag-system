"""Script to transform the original query of the user to prepare it for the search."""

from typing import Protocol

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import SecretStr

from src.config import config


class QueryTransformer(Protocol):
    """Interface for a query transformer."""

    def transform(self, query: str) -> str:
        """Transforms the original query ot the user.

        Args:
            query: Original query.

        Returns:
            Transformed query.
        """


class BaselineQueryTransformer:
    """Class for the baseline query transformer, which basically does not do anything:
    it returns the query as it arrived."""

    def transform(self, query: str) -> str:
        """Transforms the original query ot the user.

        Args:
            query: Original query.

        Returns:
            Transformed query.
        """

        return query


class ExpansionQueryTransformer:
    """Class for the expansion query transformer, which passes the query to an LLM to
    expand it and makes it more suitable for the search."""

    def __init__(
        self, model: str = "llama-3.3-70b-versatile", temperature: float = 0.7
    ) -> None:
        """Constructor of the class.

        Args:
            model: Name of the model to use.
            temperature: Temperature of the model.
        """

        self.llm = ChatGroq(
            model=model,
            temperature=temperature,
            api_key=SecretStr(config.groq.groq_api_key or ""),
        )
        self.system_prompt = (
            "You are an expert search optimization engine operating as a Query "
            "Expansion Transformer in a RAG pipeline. Your sole objective is to take "
            "an ambiguous, complex, or brief user query and transform it into a "
            "comprehensive set of search variations to maximize retrieval recall from "
            "a vector database.\n\n"
            "### Core Objectives:\n"
            "1. **Bridge Vocabulary Gaps:** Identify and include industry standard "
            "jargon, abbreviations, synonyms, and alternative phrasing related to the "
            "core topic.\n"
            "2. **Deconstruct Complexity:** If the user query contains multiple "
            "underlying questions, break them down into distinct, atomic sub-queries.\n"
            "### Instructions:\n"
            "- Analyze the user's original query.\n"
            "- Generate exactly 2-3 diverse search variations.\n"
            "- Ensure each variation approaches the problem from a different angle "
            "(e.g., one conceptual, one keyword-heavy, one technical/tool-specific, "
            "one phrasing it as a solution).\n"
            "- Do not add conversational filler, preambles, or explanations.\n"
            "- Output the result strictly as a raw JSON array of strings.\n\n"
            "### Examples:\n"
            "User Query: 'How do I fix memory leaks in Python?'\n"
            "Output:\n"
            "[\n"
            "Python memory leak troubleshooting and debugging techniques, \n"
            "How to detect memory growth using tracemalloc and objgraph, \n"
            "Python memory management profiles high RSS usage fix\n"
            "]\n"
            "User Query: 'What is the difference between OAuth2 and SAML?'\n"
            "Output:\n"
            "[\n"
            "OAuth2 vs SAML 2.0 comparison security architecture, \n"
            "When to use SAML instead of OAuth for enterprise SSO, \n"
            "Token based authorization vs XML assertion authentication, \n"
            "]"
        )

    def transform(self, query: str) -> str:
        """Transforms the original query ot the user.

        Args:
            query: Original query.

        Returns:
            Transformed query.

        Raises:
            RuntimeError: If there's an error when calling the llm.
        """

        messages: list[BaseMessage] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(query),
        ]
        response = self.llm.invoke(messages)

        content = response.content
        if not isinstance(content, str):
            raise RuntimeError("Error executing the model!")

        return content


class HyDEQueryTransformer:
    """Class for the HyDE query transformer, which passes the query to an LLM to
    reformulate it and make it more suitable for the search."""

    def __init__(
        self, model: str = "llama-3.3-70b-versatile", temperature: float = 0.7
    ) -> None:
        """Constructor of the class.

        Args:
            model: Name of the model to use.
            temperature: Temperature of the model.
        """

        self.llm = ChatGroq(model=model, temperature=temperature)
        self.system_prompt = (
            "You are an expert technical writer and factual reference manual "
            "generator. Your sole task is to write a highly realistic, authoritative, "
            "and concise snippet of a technical document, article, or knowledge base "
            "entry that directly answers the user's prompt.\n\n"
            "### Core Constraints:\n"
            "1. **No Conversational Filler:** Do not include any preambles, "
            "introductions, greetings, or meta-commentary (e.g., do NOT say 'Here is "
            "the information:', 'As an AI...', or 'Based on your request'). Start "
            "directly with the factual content.\n"
            "2. **Mimic Target Documentation:** Write in a formal, declarative, and "
            "academic tone. Use dense, industry-standard terminology, code snippets, "
            "or configuration examples where appropriate.\n"
            "3. **Be Specific & Assertive:** Even if you do not know the exact "
            "specific context, write a plausible, highly structured answer. Do not use "
            "hesitant language like 'It might be...' or 'Typically people...'. Speak "
            "with absolute authority.\n"
            "4. **Length:** Keep the response between 2 to 4 paragraphs (roughly "
            "150-250 words). Focus heavily on the exact technical mechanics of the "
            "solution.\n\n"
            "### Examples:\n"
            "User Query: 'How do I configure a dead letter queue in AWS SQS?'\n"
            "Output:\n"
            "To configure a Dead Letter Queue (DLQ) in Amazon Simple Queue Service "
            "(SQS), you must first designate a primary source queue and a secondary "
            "queue to act as the DLQ. Modify the source queue's attributes by defining "
            "a 'RedrivePolicy' JSON object. This policy must specify the "
            "'deadLetterTargetArn', indicating the Amazon Resource Name of the target "
            "DLQ, and a 'maxReceiveCount', which dictating the integer threshold "
            "(between 1 and 1000) of unsuccessful message deliveries before SQS "
            "automatically transitions the message to the DLQ.\n"
            "User Query: 'What causes a 413 Payload Too Large error in Nginx?'\n"
            "Output:\n"
            "The HTTP 413 Payload Too Large error occurs when a client request body "
            "exceeds the maximum size configured on the Nginx web server. By default, "
            "Nginx sets this limit to 1 Megabyte. This boundary is governed by the "
            "'client_max_body_size' directive, which can be defined within the "
            "'http', 'server', or 'location' blocks of the 'nginx.conf' file. When an "
            "incoming 'Content-Length' header surpasses this specified allocation, "
            "Nginx immediately terminates the connection and returns the 413 status "
            "code to protect host resources."
        )

    def transform(self, query: str) -> str:
        """Transforms the original query ot the user.

        Args:
            query: Original query.

        Returns:
            Transformed query.

        Raises:
            RuntimeError: If there's an error when calling the llm.
        """

        messages: list[BaseMessage] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(query),
        ]
        response = self.llm.invoke(messages)

        content = response.content
        if not isinstance(content, str):
            raise RuntimeError("Error executing the model!")

        return content
