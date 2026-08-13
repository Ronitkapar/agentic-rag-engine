# Enterprise Agentic RAG

An agentic retrieval-augmented generation app built with FastAPI, LangGraph, Streamlit, Qdrant, and external LLM tooling. The project provides:

- a backend API for retrieval, guardrails, and response generation
- a Streamlit chat UI for interacting with the system
- evaluation utilities for measuring retrieval and guardrail quality

## Features

- FastAPI backend with a `/query` endpoint
- Streamlit frontend with chat history and source previews
- Qdrant-based vector search
- reranking and retrieval pipeline
- guardrails for request filtering and safety handling
- logging and tracing support through Logfire
- evaluation scripts under `evals/`

## Project Structure

- `app/main.py` - FastAPI entrypoint
- `ui/app.py` - Streamlit UI
- `app/agents/` - graph and agent nodes
- `app/services/retrieval/` - embeddings, Qdrant search, reranking
- `app/guardrails/` - guardrail configuration and rules
- `app/gateway/` - LLM gateway client setup
- `evals/` - evaluation scripts, metrics, and datasets
- `DATA/` - source documents used for ingestion and testing

## Prerequisites

- Python 3.11+
- access to your model and gateway credentials
- a running Qdrant instance or Qdrant Cloud cluster

## Environment Variables

Create a `.env` file in the project root with the following values:

```env
GROQ_API_KEY=
GROQ_FALLBACK_API_KEY=
PORTKEY_API_KEY=
PORTKEY_CONFIG_SLUG=
QDRANT_API_KEY=
QDRANT_CLUSTER_ENDPOINT=
GEMINI_API_KEY=
LOGFIRE_TOKEN=
BACKEND_URL=http://localhost:8000
```

Notes:

- `QDRANT_CLUSTER_ENDPOINT` should point to your Qdrant cluster URL
- `BACKEND_URL` is used by the Streamlit UI to call the backend
- if you deploy the UI on Streamlit Community Cloud, set `BACKEND_URL` in Streamlit secrets

## Local Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. Create your `.env` file.
5. Start the backend:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

6. In a second terminal, start the Streamlit UI:

```bash
streamlit run ui/app.py
```

## Docker Setup

This repo also includes Docker support.

Build and run both services:

```bash
docker compose up -d --build
```

Backend:
- `http://localhost:8000`

Frontend:
- `http://localhost:8501`

## Deployment Options

### Option 1: Streamlit Community Cloud for the UI

This is a good choice if you want the frontend hosted by Streamlit.

1. Push the repo to GitHub.
2. Deploy `ui/app.py` on Streamlit Community Cloud.
3. Add secrets in the Streamlit app settings.
4. Make sure `BACKEND_URL` points to a public backend URL.

Example Streamlit secret:

```toml
BACKEND_URL = "https://your-backend-domain.com"
LOGFIRE_TOKEN = "your-logfire-token"
```

### Option 2: AWS EC2 with Docker

This repo can also be deployed on a single EC2 instance.

1. Launch an Ubuntu EC2 instance.
2. Install Docker and Docker Compose.
3. Copy the repo to the server.
4. Add the `.env` file.
5. Run:

```bash
docker compose up -d --build
```

6. Put Nginx in front if you want a clean domain and HTTPS.

## Evaluation

The `evals/` folder contains scripts and datasets for evaluating:

- retrieval quality
- guardrail behavior
- response quality

Use these utilities if you want to validate changes before shipping updates.

## Notes

- The app relies on external APIs, so missing credentials will cause runtime failures.
- If the Streamlit UI cannot reach the backend, check `BACKEND_URL`.
- If retrieval fails, check your Qdrant URL, API key, and collection name.

## License

No license has been specified yet.
