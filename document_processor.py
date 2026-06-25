import io
from PyPDF2 import PdfReader

def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extracts text from TXT, MD, or PDF files."""
    text = ""
    if filename.endswith(('.txt', '.md')):
        text = file_bytes.decode('utf-8')
    elif filename.endswith('.pdf'):
        pdf_file = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Splits a string into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks