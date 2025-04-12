from unittest.mock import patch
import os

from main import PDFPipeline

def test_pdf_pipeline_initialization(mock_config, tmp_path):
    """Test that PDFPipeline initializes correctly with config."""
    # Construct the expected absolute path using tmp_path
    expected_abs_path = tmp_path / mock_config["storage"]["save_path"]

    # Ensure the directory exists for the test
    os.makedirs(expected_abs_path, exist_ok=True)


    # Patch load_config to provide the mock config
    with patch('main.load_config', return_value=mock_config):
        # Pass tmp_path as a string
        pipeline = PDFPipeline(root_path=str(tmp_path), batch_size=5)

    assert pipeline.batch_size == 5
    assert pipeline.db_path == mock_config["database"]["arxiv_db_path"]
    assert pipeline.paper_path == mock_config["storage"]["save_path"]

    # Assert against the dynamically generated path based on tmp_path
    # Compare Path objects directly
    assert pipeline.paper_abs_path == expected_abs_path

    # You might not need to mock makedirs anymore if you create it explicitly
    # Or, assert it was called with the expected_abs_path
    # mock_makedirs.assert_called_with(str(expected_abs_path), exist_ok=True)

def test_get_unprocessed_papers(mock_pdf_pipeline_dependencies, mock_config):
    """Test retrieving unprocessed papers from database."""
    mock_cursor = mock_pdf_pipeline_dependencies['cursor']
    mock_cursor.description = [('paper_id',), ('pdf_path',), ('pdf_url',)]
    mock_cursor.fetchall.return_value = [
        ('paper1', 'path1.pdf', 'url1'),
        ('paper2', 'path2.pdf', 'url2')
    ]
    
    with patch('main.load_config', return_value=mock_config):
        pipeline = PDFPipeline()
        papers = pipeline._get_unprocessed_papers()
        
        assert len(papers) == 2
        assert papers[0]['paper_id'] == 'paper1'
        assert papers[1]['paper_id'] == 'paper2'
        mock_cursor.execute.assert_called_once()

def test_ensure_pdf_exists_with_existing_pdf(mock_pdf_pipeline_dependencies, mock_config):
    """Test PDF existence check when PDF already exists."""
    mock_cursor = mock_pdf_pipeline_dependencies['cursor']
    
    # Test with a paper that has a pdf_path
    paper = {'paper_id': 'test1', 'pdf_path': 'existing/path.pdf'}
    
    with patch('main.load_config', return_value=mock_config):
        pipeline = PDFPipeline()
        pdf_path = pipeline._ensure_pdf_exists(paper)
        
        assert pdf_path == 'existing/path.pdf'
        # Verify that no database query was made since pdf_path exists
        mock_cursor.execute.assert_not_called()

def test_ensure_pdf_exists_with_missing_pdf(mock_pdf_pipeline_dependencies, mock_config):
    """Test PDF existence check when PDF needs to be downloaded."""
    mock_cursor = mock_pdf_pipeline_dependencies['cursor']
    mock_cursor.fetchone.side_effect = [(None,), ('downloaded/path.pdf',)]
    mock_fetcher = mock_pdf_pipeline_dependencies['fetcher']
    
    # Test with a paper that has no pdf_path but has a pdf_url
    paper = {'paper_id': 'test1', 'pdf_url': 'http://example.com/paper.pdf'}
    
    with patch('main.load_config', return_value=mock_config):
        pipeline = PDFPipeline()
        pdf_path = pipeline._ensure_pdf_exists(paper)
        
        assert pdf_path == 'downloaded/path.pdf'
        # Verify database operations for missing PDF
        mock_cursor.execute.assert_called_with(
            "SELECT pdf_path FROM papers WHERE paper_id = ?",
            ('test1',)
        )
        mock_fetcher.download_arxiv_pdf.assert_called_once()

def test_process_paper_success(mock_pdf_pipeline_dependencies, mock_config):
    """Test successful processing of a single paper."""
    mock_cursor = mock_pdf_pipeline_dependencies['cursor']
    mock_faiss = mock_pdf_pipeline_dependencies['faiss']
    
    paper = {'paper_id': 'test1', 'pdf_path': 'test.pdf'}
    
    with patch('main.load_config', return_value=mock_config):
        pipeline = PDFPipeline()
        pipeline._process_paper(paper)
        
        # Verify database update
        mock_cursor.execute.assert_called_with(
            """
                    UPDATE papers 
                    SET text_extracted = 1 
                    WHERE paper_id = ?
                """,
            ('test1',)
        )
        
        # Verify FAISS index update
        mock_faiss.insert_chunks_into_db.assert_called_once()

def test_process_pdfs_empty_batch(mock_pdf_pipeline_dependencies, mock_config):
    """Test processing when no unprocessed papers are available."""
    mock_cursor = mock_pdf_pipeline_dependencies['cursor']
    mock_cursor.description = [('paper_id',), ('pdf_path',), ('pdf_url',)]
    mock_cursor.fetchall.return_value = []
    
    with patch('main.load_config', return_value=mock_config):
        pipeline = PDFPipeline()
        pipeline.process_pdfs()
        
        # Verify no processing occurred
        mock_cursor.execute.assert_called_once()
        mock_faiss = mock_pdf_pipeline_dependencies['faiss']
        mock_faiss.insert_chunks_into_db.assert_not_called()