import os
import streamlit as st
import requests
import time
import uuid
import logfire
from dotenv import load_dotenv


# Load environment variables explicitly from the root directory
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=env_path)


# Initialize Logfire
try:
    token = os.getenv("LOGFIRE_TOKEN")
    if not token:
        print("ERROR: LOGFIRE_TOKEN is empty or None!")
    logfire.configure(token=token)
    # logfire.instrument_requests() # Disabled due to OpenTelemetry bug on Windows: MeterProvider.get_meter() got multiple values for argument 'version'
    LOGFIRE_STATUS = "Connected & Tracing"
except Exception as e:
    print(f"Logfire Init Error in UI: {e}")
    LOGFIRE_STATUS = f"Standby (Error: {e})"
    


# PAGE CONFIG
st.set_page_config(
    page_title="Agentic RAG Engine",
    page_icon="🤖",
    layout="wide",
)

# emojie
AI_AVATAR = "🤖"
USER_AVATAR = "👤"


# SESSION MANAGEMENT 
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logfire.info(f"✨ New User Session Created: {st.session_state.session_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []


# LANDING PAGE 
# Every capability listed here is implemented in this repo (agent graph,
# guardrails, Qdrant + FlashRank retrieval, MemorySaver, ingestion loaders,
# RAGAS eval pipeline). Keep the list in sync with the code: the README
# previously advertised a Prometheus /metrics endpoint, API-key auth and
# LangSmith tracing that app/ never implemented, and those claims have been
# removed from it — don't reintroduce them here.
_CAPABILITIES = [
    (
        "🧭 Agentic planning",
        "A LangGraph planner reads the conversation *before* anything is "
        "searched, then routes the turn: search the documents, answer from "
        "memory, or decline.",
    ),
    (
        "🎯 Vector search + reranking",
        "Qdrant returns 8 candidate chunks; a local FlashRank cross-encoder "
        "rescores them and keeps the top 5, so noise is dropped before the "
        "answer is written.",
    ),
    (
        "🛡️ Guardrails on every query",
        "NeMo Guardrails, plus a regex pre-filter, block off-topic, "
        "prompt-injection and malicious-access requests before retrieval runs.",
    ),
    (
        "🧠 Conversational memory",
        "A session-scoped checkpointer keeps follow-up questions in context — "
        "and says so when a server restart has cost it that memory.",
    ),
    (
        "🔍 Visible reasoning & sources",
        "The chat streams the agent's step-by-step plan and the exact chunks "
        "retrieved, so you can check the evidence behind every answer.",
    ),
    (
        "📄 Local document ingestion",
        "PDF, HTML, TXT, DOCX and PPTX are parsed and chunked on-device — no "
        "external OCR service — then embedded into Qdrant.",
    ),
]


def render_landing():
    """Project intro shown by default. Sets view='chat' when Try It is clicked."""
    # Hero — centred and width-capped so the paragraph keeps a readable measure.
    _, hero, _ = st.columns([1, 4, 1])

    with hero:
        st.markdown("### 🤖 Agentic RAG Engine")
        st.title("An agent that decides when to search — and shows its work.")
        st.markdown(
            "Ordinary retrieval-augmented generation embeds every question and "
            "hopes the top chunks are useful. **Agentic** RAG puts a decision "
            "step in front: a planner reads the conversation first and chooses "
            "between answering from memory, searching your documents, or "
            "declining. When it does search, a vector store returns candidate "
            "passages and a cross-encoder reranks them, so only genuinely "
            "relevant evidence reaches the model."
        )
        st.markdown(
            "This project wires that together end to end — orchestration, "
            "retrieval, guardrails, memory and tracing — behind a single chat "
            "window, and shows you every step it took."
        )

        if st.button("🚀 Try It", type="primary", width="stretch"):
            st.session_state.view = "chat"
            st.rerun()

        st.caption('e.g. "How do I autoscale pods?" · "What does a CronJob do?"')

    st.markdown("---")
    st.subheader("What it actually does")

    # 3×2 grid; Streamlit stacks the columns vertically on narrow screens.
    for row_start in (0, 3):
        for col, (heading, body) in zip(st.columns(3), _CAPABILITIES[row_start:row_start + 3]):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{heading}**")
                    st.caption(body)

    # The eval suite is a real part of the project but a visitor never meets it
    # inside the chat, so it gets its own callout rather than a grid cell.
    with st.container(border=True):
        st.markdown("**📊 RAGAS evaluation suite**")
        st.caption(
            "Faithfulness, Answer Relevancy, Context Precision, Context Recall and "
            "Answer Correctness, plus a zero-LLM Jaccard tool-correctness check. It "
            "judges through a separate key so eval traffic never drains the "
            "production budget."
        )

    st.markdown("---")
    st.caption(
        "Stack · LangGraph · Groq gpt-oss-120b / gpt-oss-20b · Qdrant · "
        "FlashRank · NeMo Guardrails · Logfire · RAGAS evaluations · Streamlit"
    )
    st.caption("Domain scope · Kubernetes · Intel hardware · Enterprise networking")


# VIEW ROUTING 
# 'view' lives in session state so the selected page survives reruns. The
# chat below is left byte-identical; st.stop() simply prevents it from
# running while the landing page is showing.
if "view" not in st.session_state:
    st.session_state.view = "landing"

if st.session_state.view == "landing":
    render_landing()
    st.stop()

with st.sidebar:
    if st.button("← Back to Home", width="stretch"):
        st.session_state.view = "landing"
        st.rerun()


# SIDEBAR 
with st.sidebar:
    st.title("🧠 Agent OS")
    st.markdown("---")
    st.success(f"Logfire: {LOGFIRE_STATUS}")
    st.info(f"Memory ID: {st.session_state.session_id[:8]}")
    
    if st.button("🗑️ Clear History & Memory", width="stretch", type="primary"):
        logfire.warn(f"🗑️ Memory Wipe Triggered for session: {st.session_state.session_id}")
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# MAIN CHAT 
st.title("🤖 Enterprise Agentic Assistant")


# Display history
for message in st.session_state.messages:
    avatar = AI_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask about your documentation..."):
    # START TRACE: User Interaction
    with logfire.span("💬 User Chat Interaction", user_query=prompt, session_id=st.session_state.session_id):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        # Assistant Response
        with st.chat_message("assistant", avatar=AI_AVATAR):
            with st.status("🔍 Agent is thinking...", expanded=True) as status:
                # requests.post below is BLOCKING, so a spinner here would freeze
                # rather than animate. Write the expectation into the status body
                # instead, so a slow first query reads as "working" and not "hung".
                st.caption(
                    "⏳ Waiting for the backend. On a free-tier host the first "
                    "request after ~15 min idle pays a cold start (imports + guard "
                    "init) and can take 1–3 minutes."
                )
                try:
                    # DISTRIBUTED TRACE: Calling Backend
                    with logfire.span("📡 Calling RAG Backend"):
                        # Get backend URL from env, or default to local if not set
                        base_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                        url = f"{base_url}/query"
                        # 60s was sized for a warm backend. On Render's free tier
                        # the first /query after a spin-down routinely exceeds that
                        # on its own (README quotes 30-90s just for the Python
                        # import at 0.1 vCPU, before any LLM call), and the UI
                        # service can itself be waking from its own spin-down at
                        # the same moment. 300s covers two chained cold starts;
                        # exceeding it still raises cleanly into the except below.
                        payload = {
                            "q": prompt,
                            "thread_id": st.session_state.session_id,
                            # Report how many prior turns THIS client holds, so the
                            # server can tell a genuinely new thread from one whose
                            # in-process memory it lost to a restart.
                            "client_turns": len(
                                [m for m in st.session_state.messages if m["role"] == "user"]
                            ) - 1,
                        }
                        response = requests.post(url, json=payload, timeout=300)
                        data = response.json()
                    
                    # Show Reasoning Steps from Backend
                    steps = data.get("thought_process", [])
                    for step in steps:
                        st.write(f"⚙️ {step}")
                    
                    status.update(label="✅ Answer Synthesized", state="complete", expanded=False)
                    
                    # --- SHOW SOURCES (NESTED EXPANDABLES) ---
                    sources = data.get("sources", [])
                    if sources:
                        with st.expander("📄 View Retrieved Context (Sources)"):
                            for i, source in enumerate(sources):
                                # Create a preview title for each chunk
                                preview = source[:100].replace("\n", " ") + "..."
                                with st.expander(f"Chunk {i+1}: {preview}"):
                                    st.info(source)
                except Exception as e:
                    logfire.error(f"❌ UI-Backend Connection Failed: {e}")
                    status.update(label="❌ Connection Failed", state="error")
                    st.error("Backend Offline.")
                    st.stop()

            # Final Answer Streaming
            answer_placeholder = st.empty()
            full_answer = data.get("answer", "No response.")
            
            curr_text = ""
            for char in full_answer:
                curr_text += char
                answer_placeholder.markdown(curr_text + "▌")
                time.sleep(0.005)
            
            answer_placeholder.markdown(full_answer)
            st.session_state.messages.append({"role": "assistant", "content": full_answer})
            logfire.info("✅ Chat cycle completed successfully.")
