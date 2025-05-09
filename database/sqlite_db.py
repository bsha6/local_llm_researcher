import sqlite3
import logging
from typing import Optional
from utils.file_operations import load_config


class DatabaseManager:
    _config = None

    @classmethod
    def get_config(cls):
        if cls._config is None:
            cls._config = load_config()
        return cls._config

    def __init__(self, db_path: Optional[str] = None):
        config = self.get_config()
        self.db_path = db_path or config["database"]["arxiv_db_path"]

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        return self.conn.cursor()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.conn.commit()
        self.conn.close()
    
    def init_db(self):
        """Creates the SQLite database tables if they don't exist."""
        with self as cursor:
            # Create papers table
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
                downloaded INTEGER DEFAULT 0, -- 0 means FALSE
                text_extracted INTEGER DEFAULT 0 -- 0 means FALSE
            )
            """)
            
            # Create query history table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                query_id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                max_results INTEGER NOT NULL,
                results_returned INTEGER,
                category TEXT,  -- Optional category/topic of the search
                status TEXT DEFAULT 'completed'  -- Track query execution status
            )
            """)

            # Create paper chunks table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                faiss_idx INTEGER,
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
            );
            """)
            
            logging.info("Database tables initialized successfully")
    
    @staticmethod
    def _fetch_metadata(db_path, query, params=None):
        """Fetch metadata from the database using the DatabaseManager context."""
        with DatabaseManager(db_path) as cursor:
            cursor.execute(query, params or ())  # Handles None case for params
            return cursor.fetchall()  # Returns the query results


if __name__ == "__main__":
    db = DatabaseManager()
    db.init_db()