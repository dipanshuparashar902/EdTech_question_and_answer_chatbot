import os
import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

# ---------- Page config ----------
st.set_page_config(
    page_title="Codebasics Q&A",
    page_icon="🌱",
    layout="wide",
)

# ---------- Custom CSS ----------
st.markdown(
    """
    <style>
        /* Global page */
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

        /* Floating card */
        .floating-card {
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

        /* Button */
        .stButton > button {
            background: #22c55e;
            color: white;
            border-radius: 999px;
            border: none;
            padding: 0.45rem 1.3rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            box-shadow: 0 12px 25px rgba(34, 197, 94, 0.35);
        }
        .stButton > button:hover {
            background: #16a34a;
            box-shadow: 0 16px 30px rgba(34, 197, 94, 0.5);
        }

        /* Text input */
        .stTextInput > div > div > input {
            background-color: #f9fafb;
            border-radius: 999px;
            border: 1px solid #d1d5db;
            color: #111827;
        }

        /* Answer box */
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

        /* Expander (source FAQs) */
        .streamlit-expanderHeader {
            font-size: 0.9rem;
            color: #111827;
        }

        .footer-text {
            text-align: center;
            margin-top: 1.2rem;
            font-size: 0.8rem;
            color: #9ca3af;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Layout container ----------
st.markdown('<div class="floating-card">', unsafe_allow_html=True)

st.markdown('<div class="title">Codebasics Q&A Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Ask anything about the Codebasics FAQ knowledge base and get grounded answers powered by Gemini.</div>',
    unsafe_allow_html=True,
)

# Top row: knowledgebase button + status
col1, col2 = st.columns([1, 2])

with col1:
    btn = st.button("Create / Refresh Knowledgebase")

with col2:
    index_dir = "faiss_index"
    index_exists = os.path.exists(os.path.join(index_dir, "index.faiss"))
    if index_exists:
        st.success("Knowledgebase loaded ✔️")
    else:
        st.info("Knowledgebase not created yet. Click the button to build it.")

if btn:
    with st.spinner("Building vector database from FAQs..."):
        create_vector_db()
    st.success("Knowledgebase created/updated successfully!")
    index_exists = True

st.markdown("---")

# Question input
question = st.text_input("Ask your question here 👇")

# Answer section
if question:
    if not index_exists:
        st.warning("Please create the knowledgebase first by clicking the button above.")
    else:
        with st.spinner("Thinking with Gemini..."):
            chain = get_qa_chain()
            response = chain(question)

        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        st.markdown('<div class="answer-label">Answer</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="answer-title">{question}</div>',
            unsafe_allow_html=True,
        )
        st.write(response["result"])
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("📎 Source FAQs"):
            for i, doc in enumerate(response["source_documents"], start=1):
                st.markdown(f"**Source {i}:** {doc.metadata.get('source')}")
else:
    st.caption("Tip: start with something like “Do you provide job assistance?” or “Do you offer EMI payments?”")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown(
    '<div class="footer-text">Built with LangChain, FAISS, and Gemini 3.5 Flash.</div>',
    unsafe_allow_html=True,
)