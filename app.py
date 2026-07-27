import os
import uuid
import tempfile
import streamlit as st

# Import backend modules
import db_manager
import rag_engine

# Page Configuration
st.set_page_config(page_title="DocuQuery RAG", page_icon="🤖", layout="wide")

# ==========================================
# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================

# Create a unique session ID for memory tracking if one doesn't exist
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

session_id = st.session_state["session_id"]
# ==========================================
# SIDEBAR: DOCUMENT MANAGEMENT
# ==========================================

with st.sidebar:
    st.title("📄 Document Center")
    st.markdown("Upload and manage the PDF documents for your RAG knowledge base.")
    
    # 1. File Uploader
    uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Process & Save Document", use_container_width=True):
            with st.spinner("Processing PDF and generating embeddings..."):
                # Save uploaded file temporarily to disk so PyPDFLoader can read it
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Generate a unique document ID
                doc_id = str(uuid.uuid4())[:8]
                filename = uploaded_file.name
                
                # Ingest into ChromaDB & SQLite
                chunk_count = rag_engine.ingest_document(tmp_path, doc_id, filename)
                db_manager.add_document(doc_id, filename)
                
                # Clean up temporary file
                os.remove(tmp_path)
                
                st.success(f"Successfully processed '{filename}' into {chunk_count} chunks!")
    
    st.divider()
    
    # 2. Document Selection for Filtered Queries
    st.subheader("Filter Knowledge Base")
    conn = db_manager.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT doc_id, filename FROM documents ORDER BY upload_timestamp DESC")
    docs = cursor.fetchall()
    conn.close()
    
    selected_doc_ids = []
    if docs:
        st.write("Select documents to search against:")
        for doc in docs:
            # Checkbox for each uploaded document
            if st.checkbox(f"📄 {doc['filename']}", value=True, key=doc['doc_id']):
                selected_doc_ids.append(doc['doc_id'])
    else:
        st.info("No documents uploaded yet. Upload a PDF above to get started.")
        
    st.divider()
    
    # 3. New Chat Session Button
    if st.button("Clear / New Chat", use_container_width=True):
        st.session_state["session_id"] = str(uuid.uuid4())
        st.rerun()

# ==========================================
# MAIN CHAT INTERFACE
# ==========================================

st.title("🤖 DocuQuery: RAG Assistant")
st.caption("Ask questions grounded strictly in your uploaded PDF documents.")

# Fetch past chat messages from SQLite DB
history = db_manager.get_chat_history(session_id)

# Display prior chat messages
for msg in history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["message"])
        # If there were citations saved with the assistant message, show them
        if msg["sources"]:
            with st.expander("🔍 Cited Sources"):
                for source in msg["sources"]:
                    st.write(f"- {source}")

# User Input Box
if prompt := st.chat_input("Ask a question about your uploaded documents..."):
    # Render user prompt in UI immediately
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Save user message to SQLite
    db_manager.save_chat_message(session_id, "user", prompt)
    
    # Generate bot response using the RAG Engine
    with st.chat_message("assistant"):
        with st.spinner("Searching vector database and generating answer..."):
            # Execute RAG query passing conversation history and document filters
            answer, sources = rag_engine.ask_question(prompt, history, selected_doc_ids)
            
            st.markdown(answer)
            if sources:
                with st.expander("🔍 Cited Sources"):
                    for source in sources:
                        st.write(f"- {source}")
                        
    # Save assistant response & sources to SQLite
    db_manager.save_chat_message(session_id, "assistant", answer, sources)