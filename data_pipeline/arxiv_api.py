import arxiv
import os
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from typing import List, Dict

from database.sqlite_db import DatabaseManager
from utils.file_operations import load_config
from utils.pdf_operations import compress_pdf


class ArxivPaperFetcher:
    """Class for fetching, displaying, and storing ArXiv papers."""

    def __init__(self, query=None, config=None):
        """Initialize with configuration."""
        # Use provided config or load default
        self.config = config if config is not None else load_config()
        
        # Use provided query or get from config
        self.query = query if query is not None else self.config["arxiv"]["query"]
        
        self.save_path = self.config["storage"]["save_path"]
        self.db_path = self.config["database"]["arxiv_db_path"]
        
        # Ensure save path exists
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, "arxiv"), exist_ok=True)


    def _store_query_history(self, query: str, max_results: int, results_returned: int, category: str = None) -> int:
        """
        Store query execution history in the database.
        
        :param max_results: Maximum number of results requested
        :param results_returned: Actual number of results returned
        :param category: Optional category/topic of the search
        :return: The query_id of the stored record
        """
        with DatabaseManager(self.db_path) as cursor:
            cursor.execute("""
                INSERT INTO query_history (query_text, max_results, results_returned, category)
                VALUES (?, ?, ?, ?)
            """, (query, max_results, results_returned, category))
            return cursor.lastrowid

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2))
    def fetch_arxiv_paper_data(self, query: str = None, max_results: int = 5, sort_by: str = "relevance", category: str = None, store_query: bool = True):
        """Fetches papers from ArXiv."""
        if query is None:
            query = self.query

        if sort_by.lower() == "date":
            sort_criterion = arxiv.SortCriterion.SubmittedDate
        else:
            sort_criterion = arxiv.SortCriterion.Relevance

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_criterion
        )

        papers = []
        for result in client.results(search):
            paper_id = result.entry_id.split("/")[-1]

            paper_data = {
                "id": paper_id,
                "doi": result.doi,
                "title": result.title,
                "authors": ", ".join(author.name for author in result.authors),
                "abstract": result.summary,
                "date": result.published.strftime("%Y-%m-%d"),
                "pdf_url": result.pdf_url,
                "primary_cat": result.primary_category,
                "journal": result.journal_ref,
                "source": "arxiv",
            }
            papers.append(paper_data)
        
        # Store query history after successful fetch
        if store_query:
            self._store_query_history(query, max_results, len(papers), category)
        
        return papers

    def display_papers(self, papers):
        """Display available papers for user selection."""
        if len(papers) == 0:
            raise ValueError("papers is empty")
        print("\nAvailable Papers:\n" + "-" * 50)
        for idx, paper in enumerate(papers):
            print(f"[{idx}] {paper['title']} ({paper['date']})")
            print(f"    Authors: {paper['authors']}")
            print(f"    PDF Link: {paper['pdf_url']}\n")

    def download_arxiv_pdf(self, paper_id, pdf_url, cursor):
        """Downloads a paper's PDF, saves it to disk, and updates the database with the file path."""
        pdf_filename = os.path.join(self.save_path, "arxiv", f"{paper_id}.pdf")
        # Store relative path instead of absolute
        relative_pdf_path = os.path.relpath(pdf_filename, self.save_path)

        if os.path.exists(pdf_filename):
            logging.info(f"PDF already exists: {pdf_filename}")
            return

        try:
            logging.info(f"Downloading: {pdf_url}")
            response = requests.get(pdf_url, timeout=10)
            response.raise_for_status()

            with open(pdf_filename, "wb") as f:
                f.write(response.content)
            
            compress_pdf(pdf_filename)

            # ✅ Update database with file path & mark as downloaded
            cursor.execute("""
                UPDATE papers 
                SET downloaded = 1, pdf_path = ? 
                WHERE paper_id = ?
            """, (relative_pdf_path, paper_id))

            logging.info(f"✅ PDF saved: {pdf_filename}, Database updated!")

        except requests.RequestException as e:
            logging.error(f"❌ Failed to download {pdf_url}: {e}")

    def store_papers_in_db(self, papers: List[Dict[str, str]]):
        """Stores fetched ArXiv papers into the SQLite database."""
        with DatabaseManager(self.db_path) as cursor:
            for paper in papers:
                # Check if paper is already in DB
                cursor.execute("SELECT paper_id FROM papers WHERE paper_id = ?", (paper["id"],))
                if cursor.fetchone():
                    logging.info(f"Skipping {paper['title']} (already in database)")
                    continue

                # Insert into DB
                cursor.execute("""
                    INSERT INTO papers (paper_id, doi, title, authors, abstract, date, pdf_url, primary_category, journal_ref, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    paper["id"], paper["doi"], paper["title"], paper["authors"], paper["abstract"], paper["date"],
                    paper["pdf_url"], paper["primary_cat"], paper["journal"], paper["source"]
                ))

                self.download_arxiv_pdf(paper["id"], paper["pdf_url"], cursor)

    def get_query_history(self, limit: int = None, category: str = None):
        """
        Retrieve query history from the database.
        
        :param limit: Optional limit on number of records to return
        :param category: Optional category to filter by
        :return: List of query history records
        """
        with DatabaseManager(self.db_path) as cursor:
            query = """
                SELECT query_id, query_text, timestamp, max_results, results_returned, category, status
                FROM query_history
                WHERE 1=1
            """
            params = []
            
            if category:
                query += " AND category = ?"
                params.append(category)
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            
            # Get column names from cursor description
            columns = [description[0] for description in cursor.description]
            
            # Convert results to list of dictionaries
            results = cursor.fetchall()
            return [dict(zip(columns, row)) for row in results]

    def get_query_stats(self):
        """
        Get statistics about queries executed.
        
        :return: Dictionary containing query statistics
        """
        with DatabaseManager(self.db_path) as cursor:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_queries,
                    AVG(results_returned) as avg_results,
                    SUM(results_returned) as total_results,
                    MIN(timestamp) as first_query,
                    MAX(timestamp) as last_query,
                    COUNT(DISTINCT category) as unique_categories
                FROM query_history
            """)
            
            # Get column names from cursor description
            columns = [description[0] for description in cursor.description]
            
            # Convert result to dictionary
            result = cursor.fetchone()
            return dict(zip(columns, result))


# -----------------------------
# 🔹 RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    try:
        fetcher = ArxivPaperFetcher()
        papers = fetcher.fetch_arxiv_paper_data(query="TEST reinforcement learning", sort_by="date")
        fetcher.display_papers(papers)
        # fetcher.store_papers_in_db(papers)
        # logging.info(f"✅ Successfully stored {len(papers)} papers in the database!")
    except Exception as e:
        logging.error(f"🚨 An error occurred: {e}")
