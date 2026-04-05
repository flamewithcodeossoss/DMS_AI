import pymupdf
from app.models.schemas import InputType


ALLOWED_IMAGE_MIMES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}
ALLOWED_PDF_MIMES = {"application/pdf"}


def detect_input_type(content_type: str) -> InputType:
    """Determine input type from MIME type."""
    if content_type in ALLOWED_PDF_MIMES:
        return InputType.PDF
    if content_type in ALLOWED_IMAGE_MIMES:
        return InputType.IMAGE
    return InputType.TEXT


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from a PDF file using pymupdf."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page_num, page in enumerate(doc, 1):
        text = page.get_text()
        if text.strip():
            pages.append(f"--- Page {page_num} ---\n{text.strip()}")
    doc.close()
    if not pages:
        return ""
    return "\n\n".join(pages)
