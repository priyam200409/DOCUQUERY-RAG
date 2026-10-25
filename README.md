🤖 DocuQuery: Multi-Document RAG Assistant

DocuQuery is an end-to-end conversational Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, LangChain, and ChromaDB. It processes PDFs using local HuggingFace embeddings and leverages Google Gemini for grounded, multi-turn QA. Features include persistent session memory, SQLite metadata tracking, document filtering, and precise source citations.

DEPLOYED LINK - https://docuquery-rag-assistant.streamlit.app/

✨ Key Features
Multi-Document Knowledge Base: Upload multiple PDFs and maintain a persistent local vector database

Conversational Memory: Utilizes a history-aware retriever to understand context and pronouns across multi-turn conversations.

Targeted Retrieval: Filter the knowledge base to query specific documents using SQLite metadata injection.

Source Citations: Every generated response includes exact page numbers and filenames to eliminate silent hallucinations.

Robust Fallback Architecture: Automatically cycles through active Google Gemini models to bypass regional or API-version rate limits.

Local Embeddings: Uses free, on-device Hugging Face embeddings to reduce API costs and increase ingestion speed.

🛠️ Tech Stack & Architecture
Frontend: Streamlit

Orchestration: LangChain (langchain-classic, LCEL)

Generative AI (LLM): Google Gemini API (langchain-google-genai)

Embeddings: HuggingFace Sentence Transformers (all-MiniLM-L6-v2)

Vector Database: ChromaDB (Persistent Disk Mode)

Relational Database: SQLite (Tracks document states and chat session histories)

Document Processing: PyPDFLoader, RecursiveCharacterTextSplitter



📂 Project Structure


docuquery-rag/
│
├── app.py                 # Streamlit UI, Session State, and layout mapping
├── rag_engine.py          # Core LangChain pipelines (Ingestion & QA Chains)
├── db_manager.py          # SQLite helper for metadata and history tracking
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (API Keys)
│
├── data/
│   └── chroma_db/         # Local persistent vector store (Auto-generated)
└── app_data.db            # SQLite relational database (Auto-generated)




🚀 Installation & Setup


1. Clone the repository

Bash
git clone https://github.com/priyam200409/docuquery-rag.git
cd docuquery-rag
2. Create a virtual environment

Bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
3. Install dependencies

Bash
pip install -r requirements.txt
4. Configure Environment Variables
Create a .env file in the root directory and add your Google AI Studio API key:

Plaintext
GOOGLE_API_KEY=your_google_gemini_api_key_here
💻 Usage
Start the Streamlit application by running:

Bash
streamlit run app.py

Navigate to http://localhost:8501 in your browser.

Use the Document Center sidebar to upload one or more PDF files.

Check or uncheck specific documents to filter your search context.

Ask questions in the main chat interface and view expanding citations for every grounded answer.

🧠 Core Engineering Decisions
Why Local Embeddings? Offloading embeddings to local CPU/GPU via sentence-transformers prevents hitting strict API rate limits during the batch-processing phase of heavy PDFs.

Why SQLite + ChromaDB? Vector databases are excellent for semantic search but poor at state management. SQLite bridges this gap by persisting conversation histories and mapping unique doc_ids to files.

Two-Prompt Memory Concept: Instead of blindly passing chat history to the LLM, the architecture first uses an LLM to rewrite user queries into standalone questions, resulting in vastly superior vector retrieval for follow-up queries.

Author: Priyam Srivastava
Undergraduate specializing in Artificial Intelligence & Machine Learning (AIML).
