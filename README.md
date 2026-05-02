# RAG Knowledge Tutor

A conversational AI tutoring system built with LangChain, OpenAI, and Streamlit. Upload a PDF or DOCX learning material, then ask questions — the system retrieves the most relevant passages and generates grounded, pedagogically-aware answers.

---

## Features

- **Document upload** — supports PDF and DOCX files via a drag-and-drop UI
- **Hybrid retrieval** — combines semantic vector search (Chroma) with keyword-based BM25 for higher recall
- **Cross-encoder reranking** — reranks retrieved chunks using a BAAI cross-encoder for precision
- **Query rewriting** — reformulates follow-up questions into standalone queries using conversation history
- **Intent-aware prompting** — detects query type (explanation, example, exercise, interview prep) and selects a matching prompt template
- **Strict grounding** — answers are constrained to the uploaded material; no hallucinated facts
- **Conversation memory** — multi-turn chat with context carried across turns
- **Debug panel** — inspect retrieved chunks, rerank scores, query rewrites, and detected intent

---

## Architecture

```
User Question
     │
     ▼
┌─────────────────┐
│ Query Classifier│  rule-based intent detection
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Query Rewriter  │  GPT-4o-mini + last 3 turns of history
└────────┬────────┘
         │
         ├──────────────────────┐
         ▼                      ▼
┌─────────────────┐   ┌─────────────────┐
│  Vector Search  │   │   BM25 Search   │
│  (Chroma, k=5)  │   │   (rank_bm25)   │
│  weight = 0.6   │   │   weight = 0.4  │
└────────┬────────┘   └────────┬────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
         ┌─────────────────────┐
         │  EnsembleRetriever  │  fused ranked list
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │  Cross-Encoder      │  BAAI/bge-reranker-base
         │  Reranker (top 3)   │
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │  Prompt Selection   │  5 templates by intent
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │  GPT-4o-mini (RAG)  │
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │  Answer + Sources   │
         └─────────────────────┘
```

---

## Project Structure

```
RAG_Model/
├── app.py                        # Streamlit entry point
├── requirements.txt
├── .env                          # OPENAI_API_KEY (never commit this)
├── docs/                         # Place documents to ingest here
├── vectorstore/                  # Chroma persistent vector database
└── src/
    ├── ingestion/
    │   ├── loader.py             # Load PDF / DOCX from docs/
    │   └── ingest.py             # Chunk & embed into Chroma
    ├── retrieval/
    │   ├── retriever.py          # Hybrid vector + BM25 retriever
    │   ├── rag_chain.py          # Full RAG orchestration pipeline
    │   ├── query_rewriter.py     # Contextual query rewriting
    │   └── reranker.py           # Cross-encoder reranking
    ├── prompts/
    │   └── prompt_templates.py   # 5 intent-specific prompt templates
    ├── utils/
    │   ├── config.py             # All tuneable settings
    │   └── query_classifier.py   # Rule-based intent detection
    ├── evaluation/
    │   └── sql_grader.py         # LLM-based SQL answer grader
    └── test.py                   # CLI test harness
```

---

## Setup

### Prerequisites

- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/account/api-keys)

### Install

```bash
# 1. Clone the repo
git clone <repo-url>
cd RAG_Model

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

---

## Usage

### Web UI (Streamlit)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

1. **Upload** a PDF or DOCX file using the file uploader.
2. Click **"Process Document"** to chunk and index the content.
3. Type a question in the text box and click **"Ask"**.
4. The answer appears with source citations. Expand the debug panel to inspect retrieved chunks and scores.

### CLI

```bash
python src/test.py
```

Type questions at the prompt; the pipeline runs the same RAG chain and prints the answer with source excerpts.

---

## Configuration

All settings live in [src/utils/config.py](src/utils/config.py):

| Setting | Default | Description |
|---|---|---|
| `CHAT_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between adjacent chunks |
| `VECTOR_K` | `5` | Chunks retrieved by vector search |
| `BM25_K` | `5` | Chunks retrieved by BM25 |
| `ENSEMBLE_WEIGHTS` | `[0.6, 0.4]` | Weights for [vector, BM25] |
| `USE_HYBRID_RETRIEVAL` | `True` | Toggle hybrid vs. vector-only |
| `USE_SEMANTIC_CHUNKING` | `True` | SemanticChunker vs. fixed-size |
| `DEBUG` | `True` | Show debug panel in UI |
| `STRICT_GROUNDING` | `True` | Refuse out-of-context questions |

---

## Query Intent Types

The classifier in [src/utils/query_classifier.py](src/utils/query_classifier.py) maps keywords to one of five intents, each backed by a dedicated prompt template:

| Intent | Trigger keywords | Behavior |
|---|---|---|
| `exercise` | *practice, problem, exercise* | Gives a problem to solve, withholds answer |
| `example` | *example* | Walks through a concrete worked example |
| `interview` | *interview* | Frames answer as a technical interview response |
| `explanation` | *explain, what is* | Step-by-step conceptual explanation |
| `general` | *(default)* | General tutoring assistance |

---

## How Retrieval Works

**Ingestion** — documents are split with `RecursiveCharacterTextSplitter` (or `SemanticChunker`) and embedded with `text-embedding-3-small`. Embeddings are persisted in a local Chroma database.

**Retrieval** — at query time, an `EnsembleRetriever` runs both Chroma vector search and BM25 in parallel. Results are merged with configurable weights (0.6 vector / 0.4 BM25).

**Reranking** — the merged list is scored by `BAAI/bge-reranker-base` (a cross-encoder), and the top 3 chunks by rerank score are passed to the LLM.

**Generation** — the selected prompt template + top chunks + rewritten query are sent to GPT-4o-mini. The response is grounded strictly in the retrieved text.

---

## Tech Stack

| Layer | Library |
|---|---|
| LLM & orchestration | `langchain`, `langchain-openai`, `langchain-community` |
| Vector store | `chromadb` |
| BM25 search | `rank_bm25` |
| Reranking | `sentence-transformers` (BAAI/bge-reranker-base) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Document parsing | `pypdf`, `docx2txt` |
| UI | `streamlit` |
| Token counting | `tiktoken` |

---

## Security Notes

- **Never commit your `.env` file.** It is listed in `.gitignore` but verify it is not tracked with `git status`.
- The vectorstore directory contains embeddings of your documents. Do not share it publicly if the documents are sensitive.

---

## Roadmap

- [ ] Multi-document session support
- [ ] Persistent user sessions and history
- [ ] Evaluation harness for retrieval quality (RAGAS)
- [ ] Support for additional file types (PPTX, HTML)
- [ ] SQL grader integration into the main UI
