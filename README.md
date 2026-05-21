# EdTech Question & Answer Chatbot 🎓

A question–answer assistant for an e‑learning company (Codebasics‑style) built with:

- Google Gemini (`gemini-3.5-flash`)
- LangChain (RAG pipeline)
- FAISS vector store
- Hugging Face embeddings
- Streamlit UI

Learners ask questions about courses/bootcamps, and the app answers them using a curated FAQ CSV instead of hitting support email or Discord every time.

> Note: The original tutorial used Google PaLM; this project upgrades it to Gemini and the latest modular LangChain stack.
link = https://edtechquestionandanswerchatbot.streamlit.app/
---

## 🌐 Project Overview

Edtech Q&A (e‑learning platform for data courses) receives thousands of learner questions via email and Discord. Their human team uses an internal FAQ sheet to respond.

This project builds an **LLM‑powered Q&A system** that:

- Reads the **same FAQ CSV** the staff uses  
- Creates a **vector store (FAISS)** over those FAQs  
- Exposes a **Streamlit web app** where learners can ask questions  
- Answers questions using **Gemini** with **retrieval‑augmented generation (RAG)**

### What this system does

- Reduces workload for human support by auto‑answering FAQ‑style questions  
- Keeps answers grounded in the official FAQ CSV (no hallucinated policies)  
- Gives learners near‑instant answers through a friendly web interface  

---

## 🧠 Tech Stack & Key Concepts

- **LangChain + Gemini (Google Generative AI)**
  - Uses `ChatGoogleGenerativeAI` (`gemini-3.5-flash`) as the LLM for answers
  - Integrates with LangChain’s retrieval chains / RAG patterns

- **Streamlit**
  - Frontend UI where users:
    - Build the FAQ knowledgebase
    - Ask questions
    - View answers and source FAQs

- **Hugging Face Embeddings**
  - Uses `HuggingFaceEmbeddings` (via `langchain-huggingface`)
  - Model: `hkunlp/instructor-large` for dense text embeddings

- **FAISS**
  - Vector database to store and search FAQ embeddings
  - Saved locally as `faiss_index/` and loaded at runtime

- **Modern LangChain modules**
  - `langchain_core` for prompts and Runnables
  - `langchain_community` for loaders and vector stores
  - `langchain_classic` (optional) for `RetrievalQA` compatibility
  - `langchain_google_genai` for Gemini integration
  - `langchain_huggingface` for embedding integration

---

## 🧾 Features & Updates vs Original PaLM Project

Compared to the original “Google PaLM + LangChain” Codebasics project:

-  **PaLM → Gemini upgrade**  
  `GooglePalm` is replaced with `ChatGoogleGenerativeAI` using models like `gemini-3.5-flash`.

-  **New LangChain imports**  
  Legacy imports (`langchain.document_loaders`, `langchain.embeddings`, `langchain.prompts`) are updated to:
  - `langchain_community.document_loaders`
  - `langchain_community.vectorstores`
  - `langchain_huggingface.HuggingFaceEmbeddings`
  - `langchain_core.prompts.PromptTemplate`

-  **Updated FAISS handling**
  - Uses `FAISS.save_local()` and `FAISS.load_local(..., allow_dangerous_deserialization=True)` with a `faiss_index` folder.

-  **Environment handling**
  - API key stored in `.env` as:
    ```env
    GOOGLE_API_KEY=your_gemini_api_key_here
    ```
  - Loaded using `python-dotenv` in `langchain_helper.py`.

-  **Improved UI**
  - Soft, light theme with a floating card layout
  - Clear “Create / Refresh Knowledgebase” flow
  - Answer box and collapsible “Source FAQs” section

---

## 📁 Project Structure

```text
EdTech_question_and_answer_chatbot/
├── main.py                # Streamlit app (UI)
├── langchain_helper.py    # LangChain + FAISS + Gemini logic
├── dataset/
│   └── codebasics_faqs.csv  # FAQ CSV used as knowledge base
├── faiss_index/           # Generated FAISS index (ignored in Git)
├── .env                   # Stores GOOGLE_API_KEY (ignored in Git)
├── .venv/                 # Virtual environment (ignored in Git)
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── .gitignore             # Git ignore rules
```

---

## 🖼️ Screenshots

### Home Screen
<img width="1745" height="846" alt="home2" src="https://github.com/user-attachments/assets/f1811e11-a52c-4276-ade4-8fb717be3009" />


### Example Answer
<img width="1715" height="866" alt="tc1" src="https://github.com/user-attachments/assets/6b931d91-ac04-4080-8760-84f828faf99b" />

<img width="1407" height="870" alt="tc2" src="https://github.com/user-attachments/assets/fb7cc5ca-62ff-4285-86a2-b82570f52404" />



---

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/dipanshuparashar902/EdTech_question_and_answer_chatbot.git
cd EdTech_question_and_answer_chatbot
```

2. **Create and activate a virtual environment (Windows)**

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

3. **Install dependencies**

```bash
python -m pip install -r requirements.txt
```

4. **Set up your Gemini API key**

- Go to [Google AI Studio](https://ai.google.dev/) and create an API key.
- Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

---

## 🚀 Usage

1. **Run the Streamlit app**

```bash
streamlit run main.py
```

2. **Open the app**

- Your browser should open automatically (or visit the URL shown in the terminal, usually `http://localhost:8501`).

3. **Create the knowledgebase**

- Click **“Create / Refresh Knowledgebase”**.
- Wait until you see a success message.
- A `faiss_index/` folder will be created in the project directory.

4. **Ask questions**

- Type your question in the **“Ask your question here”** box.
- Press Enter.
- The app will:
  - Retrieve relevant FAQs using FAISS
  - Build a context
  - Call Gemini to generate an answer grounded in the FAQ content
  - Show the answer and the source FAQs

---

## ❓ Sample Questions

You can try questions like:

- `Do you guys provide internship and also do you offer EMI payments?`
- `Do you have javascript course?`
- `Should I learn Power BI or Tableau?`
- `I've a MAC computer. Can I use Power BI on it?`
- `I don't see Power Pivot. How can I enable it?`

These correspond to actual entries in the FAQ CSV and demonstrate how the system pulls multiple relevant FAQs. [file:1]

---

## 🔍 How It Works (High-Level)

1. **Load FAQ CSV**
   - `CSVLoader` loads `codebasics_faqs.csv` and wraps each row as a `Document`.

2. **Generate embeddings**
   - Each FAQ document is embedded using `HuggingFaceEmbeddings("hkunlp/instructor-large")`.

3. **Build FAISS index**
   - Embeddings are stored in FAISS and saved as `faiss_index/`.

4. **Retrieve relevant FAQs**
   - At query time, the question is embedded and used to fetch the top‑k similar documents.

5. **RAG with Gemini**
   - A prompt template combines:
     - `CONTEXT` (retrieved FAQ snippets)
     - `QUESTION` (user query)
   - Gemini (`gemini-3.5-flash`) generates an answer **restricted to the provided context**.

---

## 🔐 Environment & Security

- API keys are stored only in `.env`, not in the code.
- `.env`, `.venv/`, and `faiss_index/` are excluded from Git via `.gitignore`.
- Anyone cloning the repo must provide their own `GOOGLE_API_KEY`.

---

## 📌 Future Improvements

- Add conversation history (chat-style interface)
- Support multiple FAQ CSVs (per course / product)
- Dockerization for easier deployment

---

If you’re reviewing this as part of a capstone or portfolio, this project showcases:

- End‑to‑end RAG pipeline (data → embeddings → vector store → LLM)
- Migration from older LangChain / PaLM to modern Gemini + modular LangChain
- Practical UI integration with Streamlit for real learners.
