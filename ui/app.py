import os
import sys
import streamlit as st
import time
import uuid
from dotenv import load_dotenv

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Load environment variables explicitly from the root directory
env_path = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path=env_path)

try:
    import logfire
except Exception:
    logfire = None


@st.cache_resource(show_spinner=False)
def _load_rag_agent():
    # Import lazily so Streamlit can boot even if optional runtime deps are missing.
    from app.agents.graph import rag_agent

    return rag_agent


def _get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def _get_setting(key: str, default: str = "") -> str:
    # Local dev uses .env. Streamlit Cloud uses secrets.
    return os.getenv(key) or _get_secret(key, default)


# Initialize Logfire
try:
    token = _get_setting("LOGFIRE_TOKEN", "")
    if logfire and token:
        logfire.configure(token=token)
        # logfire.instrument_requests() # Disabled due to OpenTelemetry bug on Windows: MeterProvider.get_meter() got multiple values for argument 'version'
        LOGFIRE_STATUS = "Connected & Tracing"
    else:
        LOGFIRE_STATUS = "Off"
except Exception as e:
    print(f"Logfire Init Error in UI: {e}")
    LOGFIRE_STATUS = f"Standby (Error: {e})"
    


# PAGE CONFIG
st.set_page_config(
    page_title="Enterprise Agentic RAG",
    page_icon="🤖",
    layout="wide",
)

# emojie
AI_AVATAR = "🤖"
USER_AVATAR = "👤"


# SESSION MANAGEMENT 
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    if logfire and LOGFIRE_STATUS == "Connected & Tracing":
        logfire.info(f"✨ New User Session Created: {st.session_state.session_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []


# SIDEBAR 
with st.sidebar:
    st.title("🧠 Agent OS")
    st.markdown("---")
    st.success(f"Logfire: {LOGFIRE_STATUS}")
    st.info(f"Memory ID: {st.session_state.session_id[:8]}")
    
    if st.button("🗑️ Clear History & Memory", width="stretch", type="primary"):
        if logfire and LOGFIRE_STATUS == "Connected & Tracing":
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
    chat_span = logfire.span("💬 User Chat Interaction", user_query=prompt, session_id=st.session_state.session_id) if logfire and LOGFIRE_STATUS == "Connected & Tracing" else None
    if chat_span:
        chat_span.__enter__()
    try:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        # Assistant Response
        with st.chat_message("assistant", avatar=AI_AVATAR):
            with st.status("🔍 Agent is thinking...", expanded=True) as status:
                try:
                    rag_agent = _load_rag_agent()
                    # Streamlit-only deployment:
                    # Call the agent graph directly instead of POSTing to a backend.
                    # backend_url = _get_setting("BACKEND_URL", "http://localhost:8000")
                    # response = requests.post(f"{backend_url}/query", json=payload, timeout=60)
                    initial_state = {
                        "messages": [{"role": "user", "content": prompt}],
                        "current_query": prompt,
                        "documents": [],
                        "plan": ["Start"],
                        "status": "Initializing Graph...",
                        "final_answer": "",
                    }
                    config = {"configurable": {"thread_id": st.session_state.session_id}}

                    graph_span = logfire.span("📡 Calling RAG Graph") if logfire and LOGFIRE_STATUS == "Connected & Tracing" else None
                    if graph_span:
                        graph_span.__enter__()
                    try:
                        final_output = rag_agent.invoke(initial_state, config=config)
                        data = {
                            "answer": final_output.get("final_answer", "No response."),
                            "thought_process": final_output.get("plan", []),
                            "status": final_output.get("status", "unknown"),
                            "sources": final_output.get("documents", []),
                        }
                    finally:
                        if graph_span:
                            graph_span.__exit__(None, None, None)
                    
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
                    if logfire and LOGFIRE_STATUS == "Connected & Tracing":
                        logfire.error(f"❌ UI-Graph Execution Failed: {e}")
                    status.update(label="❌ Connection Failed", state="error")
                    st.error("Agent execution failed.")
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
            if logfire and LOGFIRE_STATUS == "Connected & Tracing":
                logfire.info("✅ Chat cycle completed successfully.")
    finally:
        if chat_span:
            chat_span.__exit__(None, None, None)
