import sqlite3
import json
import os

DB_PATH = "app_data.db"

def get_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Creates the required tables if they do not exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Table 1: Track uploaded documents
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT UNIQUE,
            filename TEXT,
            upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Table 2: Track chat history per user session
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            message TEXT,
            sources TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

def add_document(doc_id: str, filename: str):
    """Registers an uploaded document in SQLite."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO documents (doc_id, filename) VALUES (?, ?)",
        (doc_id, filename)
    )
    conn.commit()
    conn.close()

def save_chat_message(session_id: str, role: str, message: str, sources: list = None):
    """Saves a user or bot message along with cited sources into SQLite."""
    conn = get_connection()
    cursor = conn.cursor()
    sources_str = json.dumps(sources) if sources else "[]"
    
    cursor.execute(
        "INSERT INTO chat_history (session_id, role, message, sources) VALUES (?, ?, ?, ?)",
        (session_id, role, message, sources_str)
    )
    conn.commit()
    conn.close()

def get_chat_history(session_id: str):
    """Fetches full chat conversation history for a given session."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, message, sources FROM chat_history WHERE session_id = ? ORDER BY id ASC",
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            "role": row["role"],
            "message": row["message"],
            "sources": json.loads(row["sources"]) if row["sources"] else []
        })
    return history

# Automatically initialize database tables on import
init_db()