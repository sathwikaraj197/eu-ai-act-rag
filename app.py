import os
import streamlit as st
import google.generativeai as genai
from rag import load_and_chunk_pdf, build_index, search

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EU AI Act Q&A Assistant",
    page_icon="⚖️",
    layout="wide",
)

# ── Load RAG system once and cache ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_rag():
    chunks = load_and_chunk_pdf("eu_ai_act.pdf")
    index, model, chunks = build_index(chunks)
    return index, model, chunks


# ── Generate answer via Gemini API ─────────────────────────────────────────────
def generate_answer(query: str, context_chunks: list[str], gemini_key: str) -> str:
    trimmed = [" ".join(c.split()[:250]) for c in context_chunks[:4]]
    context = "\n\n---\n\n".join(trimmed)
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        "You are an expert on the EU Artificial Intelligence Act. "
        "Answer the question using ONLY the context passages provided below. "
        "Be accurate and concise. Cite article numbers when they appear.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )
    response = model.generate_content(prompt)
    return response.text.strip()


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("⚖️ EU AI Act Q&A Assistant")
st.markdown(
    "Ask any question about the **EU Artificial Intelligence Act** — "
    "answers are grounded in the official regulation text."
)

# Load index
with st.spinner("📚 Loading EU AI Act knowledge base — this takes ~30 s on first load…"):
    index, embed_model, chunks = load_rag()
st.success(f"✅ Ready — {len(chunks):,} passages indexed from the EU AI Act")

# Gemini API key (from Streamlit secrets or env)
gemini_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
if not gemini_key:
    st.warning(
        "⚠️ No Gemini API key found. "
        "Set **GEMINI_API_KEY** in Streamlit Cloud secrets to enable answer generation."
    )

# Example questions in sidebar
with st.sidebar:
    st.header("💡 Example questions")
    examples = [
        "What is a high-risk AI system?",
        "What obligations do providers of high-risk AI systems have?",
        "What is prohibited under the EU AI Act?",
        "How does the EU AI Act define general-purpose AI?",
        "What are the transparency requirements for AI systems?",
        "What penalties apply for violations of the EU AI Act?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending_query = ex

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📄 Source passages"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Passage {i}:** {src[:350]}…")

# Handle sidebar button injection
if "pending_query" in st.session_state:
    pending = st.session_state.pop("pending_query")
    st.session_state.messages.append({"role": "user", "content": pending})
    st.rerun()

# Chat input
if user_query := st.chat_input("Ask about the EU AI Act…"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if not gemini_key:
            answer = "⚠️ Please add your Gemini API key to Streamlit secrets (key: `GEMINI_API_KEY`) to enable answer generation."
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            with st.spinner("🔍 Searching regulation text and generating answer…"):
                relevant = search(user_query, index, embed_model, chunks, top_k=5)
                answer = generate_answer(user_query, relevant, gemini_key)

            st.markdown(answer)
            with st.expander("📄 Source passages"):
                for i, src in enumerate(relevant, 1):
                    st.markdown(f"**Passage {i}:** {src[:350]}…")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": relevant}
            )
