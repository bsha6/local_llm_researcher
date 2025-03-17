import pytest
import subprocess
import os
from unittest.mock import patch
from utils.pdf_operations import compress_pdf

def expected_gs_command(gs_path, input_path):
    """Helper function to generate the expected Ghostscript command"""
    temp_path = input_path + ".compressed"
    return [
        gs_path,
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dPDFSETTINGS=/printer",
        "-o", temp_path, input_path
    ]

class TestCompressPDF:
    
    def test_compress_pdf_homebrew_install(self, mock_subprocess_run, mock_os_path_exists):
        """Test compress_pdf using Ghostscript installed via Homebrew."""
        input_path = "input.pdf"
        temp_path = input_path + ".compressed"

        # Ensure Homebrew Ghostscript exists
        mock_os_path_exists.side_effect = lambda path: path in {"/opt/homebrew/bin/gs", input_path, temp_path}

        with patch("os.replace") as mock_os_replace, \
             patch("os.path.getsize", side_effect=lambda x: 100 if x == input_path else 50):
            compress_pdf(input_path)

            # Ensure subprocess was called with Ghostscript
            mock_subprocess_run.assert_called_once_with(
                expected_gs_command("/opt/homebrew/bin/gs", input_path), check=True
            )

            # Ensure the compressed file is copied back to the original location
            mock_os_replace.assert_called_once_with(temp_path, input_path)

    def test_compress_pdf_finds_gs_via_shutil_which(self, mock_subprocess_run, mock_os_path_exists, mock_shutil_which):
        """Test compress_pdf when gs is found via shutil.which."""
        input_path = "input.pdf"
        temp_path = input_path + ".compressed"

        mock_os_path_exists.side_effect = lambda path: path == temp_path  # Simulate output file existence

        with patch("os.replace") as mock_os_replace, patch("os.path.getsize", side_effect=lambda x: 100 if x == input_path else 50):
            compress_pdf(input_path)

            mock_subprocess_run.assert_called_once_with(
                expected_gs_command("/mocked/gs/path", input_path), check=True
            )

            mock_os_replace.assert_called_once_with(temp_path, input_path)

    def test_compress_pdf_no_gs_found(self, mock_os_path_exists, mock_shutil_which):
        """Test compress_pdf when gs is not found anywhere."""
        input_path = "input.pdf"

        mock_os_path_exists.return_value = False  # No system paths
        mock_shutil_which.return_value = None  # No executable found

        with pytest.raises(FileNotFoundError) as excinfo:
            compress_pdf(input_path)

        assert "Ghostscript executable (gs) not found" in str(excinfo.value)

    def test_compress_pdf_subprocess_error(self, mock_subprocess_run, mock_gs_path):
        """Test compress_pdf when subprocess.run raises an error."""
        input_path = "input.pdf"

        mock_subprocess_run.side_effect = subprocess.CalledProcessError(1, "command")

        with pytest.raises(subprocess.CalledProcessError):
            compress_pdf(input_path, mock_gs_path)

    def test_compress_pdf_does_not_replace_if_larger(self, mock_subprocess_run, mock_gs_path):
        """Test that compress_pdf does not replace the original file if the compressed file is larger."""
        input_path = "input.pdf"
        temp_path = input_path + ".compressed"

        # Mock file existence checks for compressed output
        with patch("os.replace") as mock_os_replace, \
             patch("os.path.getsize", side_effect=lambda x: 50 if x == input_path else 100), \
             patch("os.path.exists", side_effect=lambda x: x in {temp_path, input_path}), \
             patch("os.remove") as mock_os_remove:  # Prevent actual deletion errors
            compress_pdf(input_path, mock_gs_path)

            # Ensure subprocess was called
            mock_subprocess_run.assert_called_once_with(
                expected_gs_command(mock_gs_path, input_path), check=True
            )

            # Ensure the compressed file is NOT copied back if it's larger
            mock_os_replace.assert_not_called()

            # Ensure os.remove is called for cleanup
            mock_os_remove.assert_called_once_with(temp_path)