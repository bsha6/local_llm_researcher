import unittest
from unittest.mock import patch, MagicMock
import pytest
from datetime import datetime
from tenacity import RetryError
from arxiv import SortCriterion

from data_pipeline.arxiv_api import ArxivPaperFetcher

class TestArxivAPI(unittest.TestCase):
    
    @patch('data_pipeline.arxiv_api.arxiv.Client')
    @patch('data_pipeline.arxiv_api.arxiv.Search')
    def test_fetch_arxiv_paper_data_success(self, mock_search, mock_client):
        """Test the basic successful case for fetching paper data"""
        # Set up mock return values
        mock_results = []
        for i in range(2):
            mock_result = MagicMock()
            mock_result.entry_id = f'http://arxiv.org/abs/2106.0{i}001'
            mock_result.title = f'Test Paper {i}'
            
            author1 = MagicMock()
            author1.name = f'Author {i}A'
            author2 = MagicMock()
            author2.name = f'Author {i}B'
            mock_result.authors = [author1, author2]
            
            mock_result.summary = f'Abstract for test paper {i}'
            mock_result.published = datetime(2023, 1, i+1)
            mock_result.pdf_url = f'http://arxiv.org/pdf/2106.0{i}001'
            mock_result.primary_category = 'cs.AI'
            mock_result.journal_ref = 'Journal of AI Research'
            mock_result.doi = None
            
            mock_results.append(mock_result)
        
        # Configure the mock client to return our mock results
        mock_client_instance = mock_client.return_value
        mock_client_instance.results.return_value = mock_results
        
        # Call the function through the class
        fetcher = ArxivPaperFetcher()
        papers = fetcher.fetch_arxiv_paper_data(max_results=2, sort_by="relevance")
        
        # Assertions
        assert len(papers) == 2
        
        # Check first paper structure
        paper = papers[0]
        assert paper['id'] == '2106.00001'
        assert paper['title'] == 'Test Paper 0'
        assert paper['authors'] == 'Author 0A, Author 0B'
        assert paper['abstract'] == 'Abstract for test paper 0'
        assert paper['date'] == '2023-01-01'
        assert paper['pdf_url'] == 'http://arxiv.org/pdf/2106.00001'
        assert paper['primary_cat'] == 'cs.AI'
        assert paper['journal'] == 'Journal of AI Research'
        assert paper['source'] == 'arxiv'
        
        # Verify the search was created with correct parameters
        mock_search.assert_called_once_with(
            query=fetcher.query,
            max_results=2,
            sort_by=mock_search.call_args[1]['sort_by']
        )
        
        # Verify sort criterion was set to Relevance
        assert mock_search.call_args[1]['sort_by'] == SortCriterion.Relevance
    
    @patch('data_pipeline.arxiv_api.arxiv.Client')
    @patch('data_pipeline.arxiv_api.arxiv.Search')
    def test_fetch_arxiv_paper_data_empty(self, mock_search, mock_client):
        """Test the case where no papers are returned"""
        # Configure the mock client to return empty results
        mock_client_instance = mock_client.return_value
        mock_client_instance.results.return_value = []
        
        # Call the function through the class
        fetcher = ArxivPaperFetcher()
        papers = fetcher.fetch_arxiv_paper_data(max_results=5, sort_by="relevance")
        
        # Assertions
        assert papers == []
    

    @patch('data_pipeline.arxiv_api.arxiv.Client')
    @patch('data_pipeline.arxiv_api.arxiv.Search')
    def test_fetch_arxiv_paper_data_sort_by_date(self, mock_search, mock_client):
        """Test that sort_by parameter correctly sets the sort criterion to date"""
        # Configure the mock client to return empty results (we're just testing the sort parameter)
        mock_client_instance = mock_client.return_value
        mock_client_instance.results.return_value = []
        
        # Call the function through the class with sort_by="date"
        fetcher = ArxivPaperFetcher()
        fetcher.fetch_arxiv_paper_data(max_results=5, sort_by="date")
        
        # Verify the search was created with correct sort parameter
        assert mock_search.call_args[1]['sort_by'] == SortCriterion.SubmittedDate
    
    @patch('data_pipeline.arxiv_api.arxiv.Client')
    def test_retry_mechanism(self, mock_client):
        """Test that fetch_arxiv_paper_data retries on failure."""
        # Configure the mock client to raise an exception
        mock_client_instance = mock_client.return_value
        mock_client_instance.results.side_effect = Exception("Mocked error")
        
        with pytest.raises(RetryError) as exc_info:
            fetcher = ArxivPaperFetcher()
            fetcher.fetch_arxiv_paper_data()

        # Extract the actual exception inside RetryError
        assert isinstance(exc_info.value.__cause__, Exception), "RetryError did not wrap an exception"
        assert "Mocked error" in str(exc_info.value.__cause__), f"Expected 'Mocked error' but got {exc_info.value.__cause__}"

        # Ensure the function was retried 3 times
        assert mock_client_instance.results.call_count == 3, f"Expected 3 retries, but got {mock_client_instance.results.call_count}"
    

    # TODO: Add test for PDF download functionality once it's properly implemented
    @pytest.mark.skip(reason="PDF download functionality is currently incomplete in the implementation")
    def test_download_arxiv_pdfs(self):
        """
        Future test for PDF download functionality.
        
        This should test:
        1. PDF downloads correctly
        3. No duplicate downloads (checks if file exists)
        4. Handles network errors when downloading
        """
        pass


if __name__ == '__main__':
    unittest.main()
