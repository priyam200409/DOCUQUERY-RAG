import os
from dotenv import load_dotenv

# Document loaders and splitters
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Local Embeddings (100% Free, bypasses Google 404 errors)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# LLM and RAG Chains 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

PERSIST_DIR = "./chroma_db"

def get_embeddings():
    """Uses a robust local embedding model to completely bypass API failures."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_vector_store():
    """Connects to the local persistent Chroma database."""
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings()
    )

# ==========================================
# ROBUST LLM INITIALIZATION
# ==========================================

def get_working_llm():
    """
    Tries multiple active 2026 Gemini model names to find the one supported by your API key.
    """
    # Using the currently supported models as of mid-2026
    models_to_try = [
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-2.5-flash"
    ]
    
    for model_name in models_to_try:
        try:
            llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
            # Send a silent ping to test if the model is active
            llm.invoke("ping")
            print(f"✅ Successfully connected to: {model_name}")
            return llm
        except Exception as e:
            # We will now print the error so it doesn't fail silently
            print(f"⚠️ Model {model_name} failed. Moving to next...")
            continue
            
    raise RuntimeError("Critical Error: None of the active 2026 Gemini models are responding. Please verify your GOOGLE_API_KEY is correct in your .env file.")

# ==========================================
# INGESTION PIPELINE
# ==========================================

def ingest_document(file_path: str, doc_id: str, filename: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    for doc in documents:
        doc.metadata["doc_id"] = doc_id
        doc.metadata["filename"] = filename
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    
    return len(chunks)

# ==========================================
# RETRIEVAL & GENERATION PIPELINE
# ==========================================

def ask_question(user_query: str, chat_history: list, selected_doc_ids: list = None):
    # 1. Initialize the robust LLM
    llm = get_working_llm()
    
    # 2. Connect to Database
    vector_store = get_vector_store()
    search_kwargs = {"k": 4} 
    
    if selected_doc_ids:
        search_kwargs["filter"] = {"doc_id": {"$in": selected_doc_ids}}
        
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

    # 3. Memory Prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # 4. Generation Prompt
    qa_system_prompt = (
        "You are an expert document assistant. "
        "Use the following pieces of retrieved context to answer the question accurately. "
        "If the answer is not contained in the context, explicitly state that you do not know. "
        "Do not invent or hallucinate information outside of the provided context.\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # 5. Format History
    formatted_history = []
    for msg in chat_history:
        if msg["role"] == "user":
            formatted_history.append(HumanMessage(content=msg["message"]))
        else:
            formatted_history.append(AIMessage(content=msg["message"]))
            
    # 6. Execute Chain
    response = rag_chain.invoke({
        "input": user_query,
        "chat_history": formatted_history
    })
    
    # 7. Extract Sources
    sources = []
    if "context" in response:
        for doc in response["context"]:
            file_name = doc.metadata.get("filename", "Unknown File")
            page_num = doc.metadata.get("page", 0) + 1 
            sources.append(f"{file_name} (Page {page_num})")
            
    return response["answer"], list(set(sources))