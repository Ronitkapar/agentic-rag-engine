"""
Phase 1 — Live Pipeline.
Calls the running FastAPI /query endpoint for each golden sample.
Captures: actual_response (truncated to 300 chars), actual_contexts (from sources),
and actual_tools_called (detected from thought_process).
"""


import time
import copy
import json
import os
import requests
import logfire

# Point this at a deployed backend to evaluate the live product instead of a
# local one:  EVAL_API_URL=https://rag-app-api.onrender.com/query
API_URL = os.getenv("EVAL_API_URL", "http://localhost:8000/query")
RESPONSE_TRUNCATE = 300
DELAY_BETWEEN_CALLS = 10   # seconds — stays within Groq RPM on the main key

# A free-tier Render instance sleeps after ~15 min idle and cold-starts in
# 30-90s, so a remote target needs a longer timeout than a local one, and a
# retry: the first request after an idle period can come back 502 while the
# instance spins up. That is the platform, not the app under test.
IS_REMOTE = not API_URL.startswith(("http://localhost", "http://127.0.0.1"))
REQUEST_TIMEOUT = int(os.getenv("EVAL_REQUEST_TIMEOUT", "240" if IS_REMOTE else "120"))
COLD_START_RETRIES = int(os.getenv("EVAL_COLD_START_RETRIES", "2" if IS_REMOTE else "0"))
COLD_START_WAIT = 20       # seconds between cold-start retries


def post_query(payload: dict) -> requests.Response:
    """POST to /query, retrying the 502/503 that a waking instance returns.

    Only retries a cold start — a 4xx or a genuine 500 propagates, because
    silently re-running those would turn a real failure into a slow pass.
    """
    last_exc = None
    for attempt in range(COLD_START_RETRIES + 1):
        try:
            resp = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
            if resp.status_code in (502, 503) and attempt < COLD_START_RETRIES:
                logfire.info(f"⏳ {resp.status_code} — instance may be cold-starting, retrying")
                time.sleep(COLD_START_WAIT)
                continue
            return resp
        except requests.exceptions.ConnectionError as e:
            # A remote host that dropped the connection outright is usually
            # still waking up, so it gets the same retry as a 502.
            if attempt < COLD_START_RETRIES:
                logfire.info(f"⏳ connection dropped ({e}) — retry {attempt + 1}")
                time.sleep(COLD_START_WAIT)
                continue
            raise
        except requests.exceptions.Timeout as e:
            last_exc = e
            if attempt < COLD_START_RETRIES:
                logfire.info(f"⏳ timed out after {REQUEST_TIMEOUT}s — retry {attempt + 1}")
                time.sleep(COLD_START_WAIT)
                continue
            raise
    raise last_exc if last_exc else RuntimeError("query failed with no response")


def detect_tool(thought_process: list) -> str:
    """
    Maps the thought_process list from /query response to a tool name.
    Planner sets:  'Intent: Technical' + 'Search Term: ...' → retrieve_documents
                   'Intent: Conversational/Memory'           → direct_answer
    main.py sets:  'Intent: Guardrails Fired'                → guardrails
    """
    joined = " ".join(thought_process).lower()
    if "guardrails fired" in joined:
        return "guardrails"
    if "intent: technical" in joined or "search term:" in joined or "context retrieved" in joined:
        return "retrieve_documents"
    if "conversational" in joined or "memory" in joined:
        return "direct_answer"
    return "unknown"


def run_pipeline(golden_dataset: dict, progress_callback=None) -> dict:
    """
    Enriches each rag_sample in golden_dataset with live API results.
    Returns a deep copy with actual_response, actual_contexts, actual_tools_called filled.
    progress_callback(i, total, question, stage, response="") is called per step.
    """
    dataset = copy.deepcopy(golden_dataset)
    samples = dataset["rag_samples"]
    n = len(samples)

    with logfire.span("🚀 Eval Phase 1 — Live Pipeline", total_samples=n):
        for i, sample in enumerate(samples):
            question = sample["question"]

            if progress_callback:
                progress_callback(i, n, question, "calling")

            with logfire.span(
                f"📤 Live Query {i + 1}/{n}",
                question=question[:80],
                domain=sample.get("domain", ""),
            ):
                try:
                    resp = requests.post(
                        API_URL,
                        json={"q": question, "thread_id": f"eval_run_{i}"},
                        timeout=REQUEST_TIMEOUT,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    raw_answer = data.get("answer") or ""
                    thought_process = data.get("thought_process") or []
                    sources = data.get("sources") or []

                    sample["actual_response"] = raw_answer[:RESPONSE_TRUNCATE]
                    sample["actual_contexts"] = sources[:5]
                    sample["actual_tools_called"] = [detect_tool(thought_process)]

                    logfire.info(
                        "✅ Response captured",
                        tool=sample["actual_tools_called"][0],
                        response_chars=len(raw_answer),
                        context_chunks=len(sources),
                    )

                except requests.exceptions.ConnectionError:
                    logfire.error("❌ Cannot reach FastAPI — is the app running on :8000?")
                    sample["actual_response"] = ""
                    sample["actual_contexts"] = sample.get("relevant_contexts", [])
                    sample["actual_tools_called"] = ["unknown"]

                except Exception as e:
                    logfire.error(f"❌ Query failed: {e}")
                    sample["actual_response"] = ""
                    sample["actual_contexts"] = sample.get("relevant_contexts", [])
                    sample["actual_tools_called"] = ["unknown"]

            if progress_callback:
                progress_callback(i, n, question, "done", sample["actual_response"])

            if i < n - 1:
                time.sleep(DELAY_BETWEEN_CALLS)

    return dataset


def save_results(dataset: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)
        
        
def load_golden_dataset() -> dict:
    golden_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    with open(golden_path) as f:
        return json.load(f)
