import sqlite3
import logging

from utils.file_operations import load_config


# Load configuration
config = load_config()

ARXIV_DB_PATH = config["database"]["arxiv_db_path"]

# -----------------------------
# 🔹 DATABASE SETUP
# -----------------------------
# TODO: add col for source (arxiv, scholars?)
def init_db():
    """Creates the SQLite database if it doesn't exist."""
    conn = sqlite3.connect(ARXIV_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS papers (
        paper_id TEXT PRIMARY KEY,  -- Unique ArXiv ID
        doi TEXT,
        title TEXT NOT NULL,  -- Paper title
        authors TEXT NOT NULL,  -- Comma-separated author list
        abstract TEXT,  -- Paper abstract
        date TEXT NOT NULL,  -- Published date (YYYY-MM-DD)
        pdf_url TEXT,  -- Direct link to PDF
        pdf_path TEXT,
        primary_category TEXT,  -- ArXiv primary category
        journal_ref TEXT,  -- Journal reference (if published)
        source TEXT,
        downloaded INTEGER DEFAULT 0 -- 0 means FALSE
        text_extracted INTEGER DEFAULT 0 -- 0 means FALSE
    )
""")
    conn.commit()
    conn.close()


class DatabaseManager:
    """Context manager for SQLite database connections."""
    
    def __init__(self, db_path):
        self.db_path = db_path

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        return self.cursor

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            logging.error(f"Database error: {exc_value}")
        self.conn.commit()
        self.conn.close()
    
    def initialize_chunk_db(self):
        """Initializes the SQLite database for storing paper chunks."""
        with DatabaseManager(self.db_path) as cursor:
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                faiss_idx INTEGER NOT NULL,
                FOREIGN KEY (paper_id) REFERENCES arxiv_papers(paper_id)
            );
            """)
        print("SQLite chunk database initialized.")

    def _fetch_metadata(db_path, query, params=None):
        """Fetch metadata from the database using the DatabaseManager context."""
        with DatabaseManager(db_path) as cursor:
            cursor.execute(query, params or ())  # Handles None case for params
            return cursor.fetchall()  # Returns the query results


if __name__ == "__main__":
    db = DatabaseManager(ARXIV_DB_PATH)
    db.initialize_chunk_db()