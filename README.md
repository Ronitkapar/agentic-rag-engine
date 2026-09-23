# Enterprise Agentic RAG

A production-ready, modular Retrieval-Augmented Generation (RAG) application combining retrieval, LLM orchestration, guardrails, and a Streamlit chat UI. Built with FastAPI, LangGraph-style agent graphs, Qdrant vector search, and pluggable LLM gateway integrations.

Purpose: provide a reusable backend for safe, explainable LLM responses grounded in indexed source documents, and a simple frontend for exploration and evaluation.

Highlights
- Full backend API for retrieval, reranking, and response generation
- Streamlit chat UI with provenance / source previews
- Qdrant-based embeddings and vector search with reranking
- Guardrails for request filtering, red-teaming, and safety handling
- Modular gateway for external LLM providers (pluggable integrations)
- Evaluation utilities for retrieval and guardrail behaviour

Quick links
- Backend entry: `app/main.py`
- Streamlit UI: `ui/app.py`
- Retrieval services: `app/services/retrieval/`
- Agent graph: `app/agents/graph.py`
- Guardrails: `app/guardrails/`
- Evaluation & metrics: `evals/`

Architecture overview

1. Ingestion & Indexing
	- Document loaders and chunking create JSON/text chunks stored as embeddings in Qdrant.

2. Retrieval Layer
	- Embedding model produces vectors for queries and documents.
	- Qdrant performs ANN search; results are optionally reranked by a reranker module to improve relevance.

3. Agent & Orchestration
	- LangGraph-style agent nodes (in `app/agents/`) implement retrieval, reasoning, and post-processing flows.
	- A gateway layer (`app/gateway/`) abstracts LLM provider calls so you can plug in different APIs.

4. Guardrails & Safety
	- Guardrail rules and policies live under `app/guardrails/` and run before/after LLM calls to enforce safety and policy constraints.

5. API & UI
	- FastAPI exposes endpoints (e.g., `/query`) consumed by the Streamlit UI or other clients.
	- Streamlit UI provides a conversational chat experience with source citations and history.

Project layout (important files)
- `app/main.py` — FastAPI application and API routes
- `app/agents/graph.py` — agent graph definition and nodes
- `app/services/retrieval/` — embeddings, Qdrant client, search & reranking logic
- `app/guardrails/` — safety rules, guardrail definitions
- `app/gateway/` — LLM gateway client and provider integrations
- `ui/app.py` — Streamlit front-end
- `evals/` — evaluation utilities, datasets, and scripts
- `DATA/` — raw documents used for ingestion/testing

Requirements & prerequisites
- Python 3.11+
- Qdrant instance or Qdrant Cloud (accessible via `QDRANT_CLUSTER_ENDPOINT`)
- API keys / credentials for any LLM provider or gateway used
- Recommended: create a virtual environment for local development

Environment variables
Create a `.env` file in the repository root with the following keys (values depend on your provider):

```env
GROQ_API_KEY=
GROQ_FALLBACK_API_KEY=
PORTKEY_API_KEY=
PORTKEY_CONFIG_SLUG=
USE_PORTKEY=false
QDRANT_API_KEY=
QDRANT_CLUSTER_ENDPOINT=
# Embeddings — backend chosen at runtime: aicredits | local
EMBEDDING_PROVIDER=aicredits
AICREDITS_API_KEY=
AICREDITS_BASE_URL=https://api.aicredits.in/v1
AICREDITS_EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_BATCH_SIZE=50
EMBEDDING_ALLOW_LOCAL_FALLBACK=true
LOGFIRE_TOKEN=
BACKEND_URL=http://localhost:8000
```

Notes:
- `QDRANT_CLUSTER_ENDPOINT` should point at your Qdrant HTTP endpoint (or leave blank to use localhost).
- `BACKEND_URL` is used by the Streamlit UI to call the backend when running the UI separately.

Local development: step-by-step

1) Create & activate a Python virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Configure `.env` with your credentials and Qdrant endpoint.

3) Start the backend (FastAPI)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4) In a new terminal, start the Streamlit UI

```bash
streamlit run ui/app.py
```

5) Open the UI at `http://localhost:8501` and the backend at `http://localhost:8000`.

Docker (recommended for reproducible runs)

```bash
# First time / after code changes (rebuild images and start):
docker compose up -d --build

# Later (images already built — just start):
docker compose up -d

# Stop:
docker compose down

# View logs:
docker compose logs -f backend
```

Service endpoints after docker-compose:
- Backend: http://localhost:8000
- Frontend (Streamlit): http://localhost:8501
- API docs (Swagger): http://localhost:8000/docs

Notes:
- If you get `permission denied while trying to connect to the docker API`, run `newgrp docker` (or log out & back in) so your terminal picks up the `docker` group membership.
- The backend has a built-in healthcheck; the frontend waits for the backend to be healthy before starting.
- Portkey is optional: set `USE_PORTKEY=true` in `.env` only when your saved Portkey config targets live models (e.g. `openai/gpt-oss-120b`). Otherwise the app calls Groq directly.

Testing & Evaluation

- The `evals/` folder contains scripts to measure retrieval accuracy, guardrail enforcement, and response quality. Use these scripts to validate changes to retrievers or guards before deploying.

Common development tasks
- Reindex data (creates the Qdrant collection if missing):

```bash
# Index just the core documents (fast):
python -m app.ingestion.processor DATA/true_data

# Index everything incl. the noisy corpus (slow, uses embedding API quota):
python -m app.ingestion.processor DATA --wipe
```

> Note: the embedding backend is chosen by `EMBEDDING_PROVIDER` (`aicredits` or
> `local`) and its vector dimension is **probed from the live provider** at startup
> (`text-embedding-3-large` → 3072-dim, `local` → 768-dim). The Qdrant collection is
> created with that probed size, and ingestion refuses to write into an existing
> collection whose size differs, telling you to re-run with `--wipe`.
>
> Switching backends is **not** only a dimension question: even when the dims match,
> the model spaces are not interchangeable. The backend that built the index is recorded
> in `processed_data/.embedding_index_meta.json` and a mismatch is logged as a warning.
> Always re-ingest with `--wipe` after changing `EMBEDDING_PROVIDER` or an embedding
> model id.

### Embedding providers (verified live)

| `EMBEDDING_PROVIDER` | Backend | Model | Dim | Notes |
|---|---|---|---|---|
| `aicredits` | api.aicredits.in `/v1/embeddings` | `text-embedding-3-large` | 3072 | **Default.** OpenAI-compatible gateway (₹ / UPI billing). Serves **only** OpenAI embedding models (`text-embedding-3-large/small/ada-002`). Every Gemini id returns `400 invalid model ID` (bare *and* `provider/`-prefixed) even though the public catalog (`/api/models`) advertises them with `supported_apis: ['embeddings']`; that route also rejects the `provider/model` prefix notation. |
| `local` | sentence-transformers | `all-mpnet-base-v2` | 768 | Offline, no API key. Also the automatic fallback when the AICredits probe fails. |

Switching between the two requires re-ingesting with `--wipe` — the dims (3072 vs 768)
and the model spaces both differ. The dim-guard and fingerprint warnings will tell you
if you forget.

Inspect the live backend at any time:

```bash
python -c "from app.services.retrieval.embedding import active_embedding_info; print(active_embedding_info())"
```

- Add a new LLM provider: extend `app/gateway/` with a provider client and register it in the gateway
- Update guardrails: modify or add rules in `app/guardrails/`

Troubleshooting
- Streamlit cannot reach backend: verify `BACKEND_URL` and CORS settings in `app/main.py`.
- Qdrant connectivity issues: verify `QDRANT_CLUSTER_ENDPOINT`, API key, and network access.
- Missing model credentials: set the appropriate environment variables and restart the services.

Security & privacy
- Secrets: keep API keys out of source control. Use environment variables, a secrets manager, or container secrets.
- Data handling: be mindful of PII in ingested documents. Add redaction or data minimization steps in the ingestion pipeline if needed.

Next steps & extension ideas
- Add automated ingestion pipelines for external content sources (S3, Google Drive, web crawlers).
- Add production-ready logging, monitoring, and tracing integrations.
- Add role-based access control on the API and UI for multi-tenant deployments.

Contributing
- Use standard Git workflow: feature branches, descriptive PR titles, and tests for new behavior.
- Run linters and tests before opening a PR.

License
- No license specified — add a LICENSE file to set the project's terms.

Contact
- For questions about this repo, open an issue or contact the maintainer.

Enjoy building with RAG!
