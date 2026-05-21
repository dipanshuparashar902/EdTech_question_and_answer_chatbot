import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA 

# Load environment variables from .env
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in .env")

# 1. Gemini LLM (using gemini-3.5-flash, which you confirmed works)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1,
)

# 2. Embeddings (modern Hugging Face integration)
embeddings = HuggingFaceEmbeddings(
    model_name="hkunlp/instructor-large"
)

# 3. Local FAISS index path
vectordb_file_path = "faiss_index"

def create_vector_db():
    """Load CSV and build FAISS index, then save it locally."""
    # Adjust this path if your CSV is somewhere else
    csv_path = os.path.join("dataset", "codebasics_faqs.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    loader = CSVLoader(
        file_path=csv_path,
        source_column="prompt",
        encoding="latin-1",  # keep if needed for special characters
    )
    data = loader.load()

    vectordb = FAISS.from_documents(
        documents=data,
        embedding=embeddings,
    )

    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    """Load FAISS index from disk and return a RetrievalQA chain."""
    # allow_dangerous_deserialization is required for load_local in newer LangChain
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
    # Quick manual test
    create_vector_db()
    chain = get_qa_chain()
    print(chain("Do you have javascript course?"))