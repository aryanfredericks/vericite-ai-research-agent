# VeriCite

An automated **literature review system** that ingests academic papers from Semantic Scholar, stores them in a hybrid vector/full-text search database, and answers research questions through a multi-agent LangGraph pipeline.

---

## How it works

```
User Query
    │
    ▼
┌─────────┐     ┌──────────┐     ┌────────────┐     ┌────────┐     ┌────────┐     ┌────────────┐
│ Planner │────▶│ Retrieval│────▶│ Aggregator │────▶│ Writer │────▶│ Critic │────▶│ Evaluation │
│ (Qwen3) │     │  (Hybrid)│     │            │     │(Llama3)│     │(Llama3)│     │ (metrics)  │
└─────────┘     └──────────┘     └────────────┘     └────────┘     └────────┘     └────────────┘
```

1. **Planner** — decomposes the user's question into focused sub-queries, each with optional metadata filters (category, year).
2. **Retrieval** — runs a hybrid search (pgvector cosine + PostgreSQL full-text) for each sub-query using Reciprocal Rank Fusion (RRF).
3. **Aggregator** — deduplicates chunks, merges them by paper, and builds a ranked context bundle.
4. **Writer** — synthesizes a literature review from the context bundle.
5. **Critic** — checks for unsupported claims, hallucinated sources, misattributions, and coverage gaps.
6. **Evaluation** — computes deterministic metrics: citation validity, source utilization, citation density, and query-draft semantic similarity.

---

## Project structure

```
.
├── app.py                          # Flask API (POST /query, GET /health)
├── docker-compose.yml              # PostgreSQL + pgvector container
├── pyproject.toml
├── .env.example                    # Environment variable reference
└── src/
    ├── agent/
    │   └── agentic_workflow.py     # LangGraph graph definition
    ├── ingestion/
    │   ├── main.py                 # CLI entrypoint (init-db, ingest, search)
    │   ├── pipeline.py             # End-to-end ingestion orchestration
    │   ├── s2_client.py            # Semantic Scholar API client
    │   ├── pdf_handler.py          # PDF download + text extraction (PyMuPDF)
    │   ├── chunker.py              # Sliding-window text chunker
    │   ├── embedder.py             # sentence-transformers embedding
    │   ├── retriever.py            # HybridRetriever (vector + lexical + RRF)
    │   ├── repository.py           # SQLAlchemy upsert / chunk replacement
    │   ├── models.py               # ORM models (papers, paper_chunks)
    │   ├── schemas.py              # Pydantic ingestion schemas
    │   ├── normalizer.py           # Raw S2 API → internal Paper model
    │   ├── db.py                   # Engine, session factory, init_db
    │   └── config.py               # Settings from environment variables
    ├── prompts/
    │   ├── planner_prompt.py
    │   ├── writer_prompt.py
    │   └── critic_prompt.py
    └── utils/
        ├── structured_outputs.py   # Pydantic models (Plan, GraphState, CriticAgentOutput)
        ├── evaluation_metrics.py   # Model-free metric functions
        ├── load_planner_llm.py
        ├── load_writer_llm.py
        └── load_critic_llm.py
```

---

## Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph + LangChain |
| LLMs | Groq — Qwen3-32b (planner), Llama-3.3-70b (writer + critic) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384 dims) |
| Vector store | PostgreSQL 17 + pgvector (HNSW index) |
| Full-text search | PostgreSQL `tsvector` / `tsquery` (GIN index) |
| API | Flask + Gunicorn |
| PDF parsing | PyMuPDF |
| Paper source | Semantic Scholar bulk search API |

---

## Setup

### 1. Start the database

```bash
docker compose up -d
```

### 2. Install dependencies

```bash
pip install uv
uv sync
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — at minimum set GROQ_API_KEY
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg2://postgres:postgres@localhost:5432/rag` | PostgreSQL connection string |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Must match at ingestion and query time |
| `EMBEDDING_DIM` | `384` | Must match the model output dimension |
| `PAPERS_PER_CATEGORY` | `100` | Papers fetched per search category |
| `OPEN_ACCESS_ONLY` | `true` | Only ingest papers with a downloadable PDF |
| `CHUNK_SIZE` | `1200` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `S2_API_KEY` | _(optional)_ | Semantic Scholar key for higher rate limits |

### 4. Initialize the database schema

```bash
uv run s2-ingest init-db
```

### 5. Ingest papers

```bash
# Ingest the default category list
uv run s2-ingest ingest

# Custom categories and limit
uv run s2-ingest ingest --categories "retrieval augmented generation" "vector databases" --limit 50
```

### 6. (Optional) Test hybrid search

```bash
uv run s2-ingest search --query "current advances in RAG systems" --top-k 5
```

---

## Running the API

**Development:**
```bash
python app.py
```

**Production:**
```bash
gunicorn -w 2 -k gthread -t 120 app:app
```

The server starts on port `8000`.

---

## API reference

### `POST /query`

Run a literature review for a research question.

**Request body:**
```json
{ "query": "What are the main approaches to improving RAG retrieval quality?" }
```

**Response:**
```json
{
  "query": "...",
  "answer": "...",
  "passed": true,
  "metrics": {
    "grounding": {
      "total_citations": 12,
      "unique_cited_sources": 5,
      "citation_validity_rate": 1.0,
      "hallucinated_source_ids": [],
      "source_utilization": 0.83,
      "papers_cited": 5,
      "papers_available": 6,
      "citation_density": 0.4
    },
    "relevance": { "query_draft_similarity": 0.87 },
    "critique": { "passed": true, "issue_score": 0 }
  },
  "critique": {
    "unsupported_claims": [],
    "missing_citations": [],
    "misattributions": [],
    "hallucinated_sources": [],
    "overgeneralizations": [],
    "coverage_gaps": [],
    "fixes": "No revisions needed.",
    "passed": true
  },
  "sources": [
    {
      "s2_paper_id": "abc123",
      "title": "...",
      "authors": ["..."],
      "year": 2024,
      "venue": "...",
      "url": "...",
      "score": 0.031
    }
  ]
}
```

### `GET /health`

```json
{ "status": "ok" }
```

---

## Evaluation metrics

The evaluation node computes three panels of metrics without calling an LLM:

**Grounding** (draft vs. retrieved context)
- `citation_validity_rate` — fraction of cited IDs that appear in the context bundle. A rate below 1.0 indicates hallucinated citations.
- `source_utilization` — fraction of retrieved papers the draft actually cites.
- `citation_density` — citations per sentence.

**Relevance** (draft vs. query)
- `query_draft_similarity` — cosine similarity between the query and the full draft using the same embedding model used at ingestion.

**Critic** (model-judged, kept separate)
- `passed` — `true` only when there are no unsupported claims, misattributions, or hallucinated sources.
- `issue_score` — weighted count of issues across all critic categories.

---

## Hybrid retrieval

The `HybridRetriever` runs two independent searches over `paper_chunks` and merges them with **Reciprocal Rank Fusion**:

- **Vector search** — HNSW cosine distance on chunk embeddings (good at paraphrase and synonyms).
- **Lexical search** — PostgreSQL `ts_rank_cd` over `content_tsv` (good at acronyms, rare terms, exact phrases).

A chunk that ranks well in either search surfaces in the results; a chunk strong in both rises to the top. RRF combines rank order rather than raw scores, so the two incomparable scales never need normalizing.

Optional JSONB containment filters (`category`, `year`) are pushed into both searches before ranking and are served by the GIN metadata index.
