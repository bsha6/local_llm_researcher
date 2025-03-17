import pytest
import unicodedata

from data_pipeline.extract.text_preprocessor import TextPreprocessor

# Fixtures for common test data
@pytest.fixture
def sample_text():
    return "   这是 一个 测试    文本。This is a TEST sentence!   "

@pytest.fixture
def multilingual_text():
    return "English text with 中文 and español"

@pytest.fixture
def long_text():
    return " ".join(["token"] * 600)

# Fixtures for processor instances
@pytest.fixture
def passage_processor(sample_text):
    return TextPreprocessor(sample_text, mode="passage")

@pytest.fixture
def query_processor(sample_text):
    return TextPreprocessor(sample_text, mode="query")

# Test initialization with different modes
def test_initialization_valid_modes():
    # Valid initialization with passage mode
    processor = TextPreprocessor("test text", mode="passage")
    assert processor.text == "test text"
    assert processor.mode == "passage"
    
    # Valid initialization with query mode
    processor = TextPreprocessor("test text", mode="query")
    assert processor.mode == "query"

def test_initialization_invalid_mode():
    # Test with invalid mode
    with pytest.raises(AssertionError):
        TextPreprocessor("test text", mode="invalid")

# Test clean_text method with different types of input
@pytest.mark.parametrize("input_text,expected", [
    ("   extra   spaces   ", "extra spaces"),
    ("UPPERCASE TEXT", "uppercase text"),
    # TODO: get this test working
    # ("café\u0301", "café"),  # é with combining accent
])
def test_clean_text(input_text, expected):
    processor = TextPreprocessor(input_text)
    assert processor.clean_text() == unicodedata.normalize("NFC", expected)

# Test multilingual handling
def test_multilingual_handling(multilingual_text):
    processor = TextPreprocessor(multilingual_text)
    processed = processor.clean_text()
    
    # Check that all languages are preserved after processing
    assert "english" in processed
    assert "中文" in processed
    assert "español" in processed
