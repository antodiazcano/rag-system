# RAG Tutorial

A modular, end-to-end **Retrieval-Augmented Generation (RAG)** system built with LangChain, Groq, Pinecone, and Sentence Transformers.

Given a PDF document, the system ingests it, chunks it, stores the embeddings in a vector database, and answers user queries by retrieving the most relevant chunks and feeding them as grounding context to an LLM.

---

## Architecture

```
flowchart TD
    subgraph IDX["📥 Indexing Pipeline (offline)"]
        PDF["📄 docs/coffee.pdf"]
        DP["<b>DocumentProcessor</b><br/><small>extract → clean → chunk → embed</small>"]
        SAVE["save_to_db()"]
    end

    PINE[("Pinecone Index")]

    subgraph QP["🔎 Query Pipeline (runtime)"]
        QUERY(["Query<br/><small>(user input)</small>"])
        QT["<b>QueryTransformer</b><br/><small>baseline / expansion / HyDE</small>"]
        SEARCH["<b>Search</b><br/><small>vector / hybrid + BM25</small>"]
        RERANK["<b>Reranker</b><br/><small>baseline / cross-encoder</small>"]
        RAG["<b>RAGPipeline</b><br/><small>orchestrator</small>"]
    end

    CHAT["<b>Chatbot</b><br/><small>Groq / Llama</small>"]
    UI["<b>Streamlit UI</b><br/><small>src/ui/app.py</small>"]

    PDF -->|"1"| DP
    DP -->|"2: chunks + embeddings"| SAVE
    SAVE -->|"3"| PINE
    QUERY -->|"4"| PINE
    QUERY -->|"5"| QT
    QUERY -->|"6"| SEARCH
    QUERY -->|"7"| RERANK
    QT -->|"8"| RAG
    SEARCH -->|"9"| RAG
    RERANK -->|"10"| RAG
    RAG -->|"11: grounding context"| CHAT
    CHAT -->|"12: answer"| UI

    classDef ingest fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#0c4a6e
    classDef store fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#78350f
    classDef query fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d
    classDef output fill:#f3e8ff,stroke:#9333ea,stroke-width:2px,color:#581c87

    class PDF,DP,SAVE ingest
    class PINE store
    class QUERY,QT,SEARCH,RERANK,RAG query
    class CHAT,UI output
```

---

## Project Structure

```
.
├── docs/
│   ├── coffee.pdf          # Source document (coffee shop manual)
│   └── coffee.txt          # Plain-text version
├── images/
│   ├── 0_rag.png
│   ├── 0_rag.excalidraw
│   ├── 1_chunking.png
│   ├── 1_chunking.excalidraw
│   ├── 2_retrieve.png
│   └── 2_retrieve.excalidraw
├── src/
│   ├── config.py           # Environment-based configuration
│   ├── utils.py            # Shared helpers (embed, Pinecone index)
│   ├── chunk/
│   │   ├── chunks.py       # DocumentProcessor, chunking strategies
│   │   └── save_to_db.py   # Upsert chunks into Pinecone
│   ├── retrieve/
│   │   ├── transformation.py  # Query expansion / HyDE strategies
│   │   ├── search.py          # Vector search & hybrid search
│   │   ├── reranking.py       # Cross-encoder reranking
│   │   ├── context.py         # Build final prompt
│   │   └── pipeline.py        # RAGPipeline orchestrator
│   ├── chatbot/
│   │   └── chatbot.py      # Conversational agent
│   └── ui/
│       └── app.py          # Streamlit web interface
├── pruebas.ipynb           # Interactive experiments
├── pyproject.toml          # Project metadata & tool config
├── Makefile                # Automation targets
├── .env.example            # Environment template
└── README.md
```

---

## Key Components

### Ingestion (`src/chunk/`)
| Component | Description |
|---|---|
| `DocumentProcessor` | Extracts text from a PDF, cleans it, and splits it into chunks using a pluggable strategy. |
| `FixedWindowChunker` | Splits text into fixed-size character windows with configurable overlap. |
| `ParagraphChunker` | Splits by paragraph boundaries (delimiters + capitalised sentences). |
| `save_to_db()` | Embeds chunks with `all-MiniLM-L6-v2` and upserts them into a Pinecone index. |

### Retrieval (`src/retrieve/`)
| Component | Description |
|---|---|
| `QueryTransformer` | Pluggable query rewriting before search: **Baseline** (no-op), **Expansion** (LLM generates search variations), or **HyDE** (generates a hypothetical document). |
| `Search` | Pluggable search: **VectorSearch** (pure embedding similarity) or **HybridSearch** (BM25 + vector with RRF fusion). |
| `Reranker` | Pluggable reranking: **Baseline** (no-op) or **CrossEncoderReranker** (cross-encoder/ms-marco-MiniLM-L-6-v2). |
| `RAGPipeline` | Orchestrates transform → search → rerank → build prompt. |

### Chatbot (`src/chatbot/`)
| Component | Description |
|---|---|
| `Chatbot` | Maintains a conversation history, sends the RAG-grounded prompt to a **Groq-hosted Llama 3.3-70B** model, and returns the answer. The system prompt enforces strict context grounding with inline citations. |

### UI (`src/ui/`)
| Component | Description |
|---|---|
| `app.py` | Streamlit chat interface with message history, expandable retrieved chunks, and a clear chat button. |

---

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)
- A **Pinecone** account (vector database)
- A **Groq** API key (LLM inference)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd rag-system

# Create .env from template and fill in your keys
cp .env.example .env

# Install dependencies
make install
```

### Environment Variables

| Variable | Description |
|---|---|
| `PINECONE_API_KEY` | Your Pinecone API key |
| `PINECONE_INDEX_NAME` | Name of the Pinecone index |
| `PINECONE_CLOUD` | Cloud provider (e.g. `aws`) |
| `PINECONE_REGION` | Region (e.g. `us-east-1`) |
| `GROQ_API_KEY` | Your Groq API key |

### Ingest the Document

```bash
python -m src.chunk.save_to_db
```

This processes `docs/coffee.pdf`, chunks it, embeds the chunks, and stores them in Pinecone.

### Run the Chatbot UI

```bash
uv run streamlit run src/ui/app.py
```

---

## Usage

1. Open the Streamlit app in your browser.
2. Type a question about the ingested document (e.g. *"What are the opening hours?"* or *"Do you have a loyalty card?"*).
3. The chatbot retrieves the most relevant chunks, grounds its answer on them, and responds with inline citations.
4. Click **"Retrieved chunks"** under any response to inspect the source material.
5. Use **"Clear chat"** to reset the conversation.
