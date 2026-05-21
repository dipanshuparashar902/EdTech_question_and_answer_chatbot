import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA  # classic RetrievalQA
from dotenv import load_dotenv

# Optional: load .env so GOOGLE_API_KEY can be used as a fallback
load_dotenv()

# Shared embeddings and FAISS index path
embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"


def create_vector_db():
    """Load CSV FAQs and build a FAISS index, then save it locally."""
    # Adjust if your CSV path is different
    csv_path = os.path.join("dataset", "codebasics_faqs.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    loader = CSVLoader(
        file_path=csv_path,
        source_column="prompt",
        encoding="latin-1",
    )
    data = loader.load()

    vectordb = FAISS.from_documents(
        documents=data,
        embedding=embeddings,
    )

    vectordb.save_local(vectordb_file_path)


def get_qa_chain(google_api_key: str | None = None):
    """
    Build and return a RetrievalQA chain.

    Priority for API key:
    1. Explicit google_api_key argument (e.g., from Streamlit sidebar)
    2. GOOGLE_API_KEY from environment / .env
    """
    if not google_api_key:
        google_api_key = os.getenv("GOOGLE_API_KEY", "")

    if not google_api_key:
        raise ValueError("Google API key not provided. "
                         "Pass it to get_qa_chain() or set GOOGLE_API_KEY in environment/.env.")

    # Load FAISS index
    vectordb = FAISS.load_local(
        vectordb_file_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

CONTEXT: {context}

QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=google_api_key,
        temperature=0.1,
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

    return chain


if __name__ == "__main__":
    # Manual test (uses GOOGLE_API_KEY from .env if google_api_key not passed)
    create_vector_db()
    chain = get_qa_chain()
    print(chain("Do you have javascript course?"))