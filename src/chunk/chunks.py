"""Script to divide a document into chunks and then create the embeddings."""

import re
from typing import Protocol

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


class ChunkingStrategy(Protocol):
    """Structural interface that any chunking strategy must satisfy."""

    def chunk(self, text: str) -> list[str]:
        """Divides the cleaned text in a list of chunks.

        Args:
            text: Cleaned text.

        Returns:
            List where each element is a chunk.
        """


class FixedWindowChunker:
    """Splits text into a fixed window of characters with overlapping."""

    def __init__(self, window_size: int = 200, overlapping: int = 25) -> None:
        """Constructor of the class.

        Args:
            window_size: Number of characters of each chunk.
            overlapping: Number of overlapping characters between chunks.
        """

        self.window_size = window_size
        self.overlapping = overlapping

    def chunk(self, text: str) -> list[str]:
        """Divides the cleaned text in a list of fixed-size chunks.

        Args:
            text: Cleaned text.

        Returns:
            List where each element is a chunk.
        """

        step = self.window_size - self.overlapping
        chunks = []

        for i in range(0, len(text), step):
            window = text[i : min(len(text) - 1, i + self.window_size)]
            chunks.append(window)

        return chunks


class ParagraphChunker:
    """Splits text into paragraphs."""

    def chunk(self, text: str) -> list[str]:
        """Divides the cleaned text in a list of paragraph chunks.

        Args:
            text: Cleaned text.

        Returns:
            List where each element is a chunk.
        """

        chunks = text.split("*")
        # Split by a line break followed by a capital letter.
        chunks = [part for chunk in chunks for part in re.split(r"\n(?=[A-Z])", chunk)]

        return chunks


class DocumentProcessor:
    """Extracts, cleans, chunks and embeds the text of a document."""

    def __init__(
        self,
        path: str,
        chunker: ChunkingStrategy,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        """Constructor of the class.

        Args:
            path: Path where the pdf is stored.
            chunker: Strategy used to split the cleaned text into chunks.
            embedding_model: Model used to embed the chunks.
        """

        self.path = path
        self.chunker = chunker
        self.embed_chunks = SentenceTransformer(embedding_model)

    def process(self) -> tuple[list[str], list[list[float]]]:
        """Processes a document to obtain chunks from it.

        Returns:
            Text of each chunk and embedding of each chunk.
        """

        raw_text = self.extract_text()
        clean_text = self.clean_text(raw_text)
        chunks = self.chunker.chunk(clean_text)
        embeddings = self.embed_chunks.encode(
            chunks, normalize_embeddings=True
        ).tolist()

        return chunks, embeddings

    def extract_text(self) -> str:
        """Extracts the whole text from the pdf.

        Returns:
            Text of the pdf.
        """

        reader = PdfReader(self.path)
        pages_text = []

        for page in reader.pages:
            text = page.extract_text()
            pages_text.append(text)

        return "\n\n".join(pages_text)

    def clean_text(self, text: str) -> str:
        """Cleans the text.

        Args:
            text: Text to clean.

        Returns:
            Cleaned text.
        """

        # Collapse more than two line breaks
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Collapse more than two spaces
        text = re.sub(r"[ \t]{2,}", " ", text)
        # Remove hyphens from the end of a line: "infor-\nmation" -> "information"
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

        return text.strip()
