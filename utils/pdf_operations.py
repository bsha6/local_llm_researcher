import subprocess
import shutil
import os


def compress_pdf(input_path, gs_path=None):
    """
    Compresses a PDF using Ghostscript.
    
    Args:
        input_pdf (str): Path to the input PDF file
        gs_path (str, optional): Custom path to the Ghostscript executable
    """
    # If gs_path is not provided, try to find Ghostscript executable
    if gs_path is None:
        # Since you installed with Homebrew, try the common Homebrew path
        if os.path.exists("/opt/homebrew/bin/gs"):
            gs_path = "/opt/homebrew/bin/gs"
        elif os.path.exists("/usr/local/bin/gs"):
            gs_path = "/usr/local/bin/gs"
        else:
            # Try to find it in PATH
            gs_path = shutil.which("gs")
            
    # If still not found, raise error
    if gs_path is None:
        raise FileNotFoundError(
            "Ghostscript executable (gs) not found. Please check your installation."
        )
    
    temp_path = input_path + ".compressed"
    command = [
        gs_path,
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dPDFSETTINGS=/printer",
        "-o", temp_path, input_path
    ]
    
    subprocess.run(command, check=True)

    # Ensure compression succeeded before replacing
    if os.path.exists(temp_path) and os.path.getsize(temp_path) < os.path.getsize(input_path):
        os.replace(temp_path, input_path)  # Atomic replacement
    else:
        os.remove(temp_path)  # Cleanup failed attempt


if __name__ == "__main__":
    # Example usage
    pdf_fp = "/Users/blake/Downloads/Complete_with_Docusign_Blake_Sha_-_Offer_Let/Blake_Sha_-_PIIA_-_Primer_Federal.pdf"
    output_path = "/Users/blake/Downloads/Complete_with_Docusign_Blake_Sha_-_Offer_Let/COMPRESSED_Blake_Sha_-_PIIA_-_Primer_Federal.pdf"
    
    # Use the function with the path to your Ghostscript executable
    # compress_pdf(pdf_fp, output_path)
    # print(f"PDF compressed successfully: {output_path}")
