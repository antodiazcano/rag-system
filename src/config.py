"""Configuration of the project."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass
class VectorDBConfig:
    """Class to define the configuration of the vector db."""

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    pinecone_cloud = os.getenv("PINECONE_CLOUD")
    pinecone_region = os.getenv("PINECONE_REGION")


@dataclass
class GroqConfig:
    """Class to define the configuration of Groq."""

    groq_api_key = os.getenv("GROQ_API_KEY")


@dataclass
class Config:
    """Main configuration class."""

    vector_db = VectorDBConfig()
    groq = GroqConfig()


config = Config()
