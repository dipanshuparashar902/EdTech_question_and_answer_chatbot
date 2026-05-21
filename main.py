import os
import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

# ---------- Page config ----------
st.set_page_config(
    page_title="EdTech Q&A Assistant",
    page_icon="🎓",
    layout="wide",
)

# ---------- Theme toggle ----------
theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)

light_css = """
<style>
    body {
        background: #f3f4f6;
        color: #111827;
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }
    .app-card {
        background: #ffffff;
        border-radius: 18px;
        padding: 2rem 2.4rem 1.8rem 2.4rem;
        box-shadow:
            0 18px 40px rgba(15, 23, 42, 0.12),
            0 0 0 1px rgba(209, 213, 219, 0.8);
        margin: 0 auto;
    }
    .title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #111827;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 0.96rem;
        margin-bottom: 1.6rem;
    }
    .stButton > button {
        background: #2563eb;
        color: white;
        border-radius: 999px;
        border: none;
        padding: 0.45rem 1.3rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        box-shadow: 0 12px 25px rgba(37, 99, 235, 0.35);
    }
    .stButton > button:hover {
        background: #1d4ed8;
        box-shadow: 0 16px 30px rgba(37, 99, 235, 0.5);
    }
    .stTextInput > div > div > input {
        background-color: #f9fafb;
        border-radius: 999px;
        border: 1px solid #d1d5db;
        color: #111827;
    }
    .answer-box {
        background: #f9fafb;
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 18px rgba(17, 24, 39, 0.06);
        margin-top: 0.4rem;
    }
    .answer-label {
        text-transform: uppercase;
        font-size: 0.72rem;
        letter-spacing: 0.16em;
        color: #10b981;
        margin-bottom: 0.25rem;
    }
    .answer-title {
        font-size: 1.02rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 0.25rem;
    }
    .footer-text {
        text-align: center;
        margin-top: 1.2rem;
        font-size: 0.8rem;
        color: #9ca3af;
    }
</style>
"""

dark_css = """
<style>
    body {
        background: radial-gradient(circle at top left, #020617, #0f172a);
        color: #e5e7eb;
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }
    .app-card {
        background: rgba(15, 23, 42, 0.9);
        border-radius: 18px;
        padding: 2rem 2.4rem 1.8rem 2.4rem;
        box-shadow:
            0 18px 40px rgba(15, 23, 42, 0.9),
            0 0 0 1px rgba(148, 163, 184, 0.25);
        margin: 0 auto;
        backdrop-filter: blur(16px);
    }
    .title {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(120deg, #a5b4fc, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 0.96rem;
        margin-bottom: 1.6rem;
    }
    .stButton > button {
        background: linear-gradient(120deg, #22c55e, #16a34a);
        color: white;
        border-radius: 999px;
        border: none;
        padding: 0.45rem 1.3rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        box-shadow: 0 12px 25px rgba(22, 163, 74, 0.4);
    }
    .stButton > button:hover {
        background: linear-gradient(120deg, #16a34a, #22c55e);
        box-shadow: 0 16px 30px rgba(22, 163, 74, 0.55);
    }
    .stTextInput > div > div > input {
        background-color: rgba(15, 23, 42, 0.9);
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.6);
        color: #e5e7eb;
    }
    .answer-box {
        background: radial-gradient(circle at top left, #0b1220, #020617);
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        border: 1px solid rgba(148, 163, 184, 0.4);
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.9);
        margin-top: 0.4rem;
    }
    .answer-label {
        text-transform: uppercase;
        font-size: 0.72rem;
        letter-spacing: 0.16em;
        color: #6ee7b7;
        margin-bottom: 0.25rem;
    }
    .answer-title {
        font-size: 1.02rem;
        font-weight: 600;
        color: #e5e7eb;
        margin-bottom: 0.25rem;
    }
    .footer-text {
        text-align: center;
        margin-top: 1.2rem;
        font-size: 0.8rem;
        color: #9ca3af;
    }
</style>
"""

if theme == "Light":
    st.markdown(light_css, unsafe_allow_html=True)
else:
    st.markdown(dark_css, unsafe_allow_html=True)

# ---------- API key input (sidebar) ----------
if "google_api_key" not in st.session_state:
    st.session_state.google_api_key = ""

st.sidebar.subheader("API Key")
st.session_state.google_api_key = st.sidebar.text_input(
    "Google API Key",
    type="password",
    help="Paste your Gemini API key. It is used only for this session. "
         "If left empty, the app will try GOOGLE_API_KEY from environment/.env.",
)

api_key = st.session_state.google_api_key

# ---------- Main card ----------
st.markdown('<div class="app-card">', unsafe_allow_html=True)

st.markdown('<div class="title">EdTech Q&A Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Ask questions about your EdTech FAQ knowledge base and get grounded answers powered by Gemini.</div>',
    unsafe_allow_html=True,
)

# ---------- Knowledgebase controls ----------
col1, col2 = st.columns([1, 2])

with col1:
    kb_btn = st.button("Create / Refresh Knowledgebase")

with col2:
    index_dir = "faiss_index"
    index_exists = os.path.exists(os.path.join(index_dir, "index.faiss"))
    if index_exists:
        st.success("Knowledgebase loaded")
    else:
        st.info("Knowledgebase not created yet. Click the button to build it.")

if kb_btn:
    with st.spinner("Building vector database from FAQs..."):
        create_vector_db()
    st.success("Knowledgebase created/updated successfully!")
    index_exists = True

st.markdown("---")

# ---------- Question input ----------
question = st.text_input("Ask your question here 👇")

# ---------- Answer section ----------
if question:
    # Resolve API key: prefer sidebar, fallback to env
    effective_api_key = api_key or os.getenv("GOOGLE_API_KEY", "")

    if not effective_api_key:
        st.warning("Please provide a Google API key in the sidebar or set GOOGLE_API_KEY in your environment.")
    elif not index_exists:
        st.warning("Please create the knowledgebase first by clicking the button above.")
    else:
        with st.spinner("Thinking with Gemini..."):
            chain = get_qa_chain(effective_api_key)
            response = chain(question)

        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        st.markdown('<div class="answer-label">Answer</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="answer-title">{question}</div>',
            unsafe_allow_html=True,
        )
        st.write(response["result"])
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Source FAQs"):
            for i, doc in enumerate(response["source_documents"], start=1):
                st.markdown(f"**Source {i}:** {doc.metadata.get('source')}")
else:
    st.caption("Tip: start with questions like “Do you provide internship and EMI?” or “Should I learn Power BI or Tableau?”")

st.markdown('</div>', unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown(
    '<div class="footer-text">Built for EdTech: LangChain · FAISS · Hugging Face · Gemini 3.5 Flash.</div>',
    unsafe_allow_html=True,
)