from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.core.config import get_settings
from app.models.schemas import (
    AnalyzeResponse,
    QueryResponse,
    QueryRequest,
    HealthResponse,
    ErrorResponse,
    InputType,
    ClinicalInsights,
)
from app.services.extractor import detect_input_type, extract_text_from_pdf
from app.services.gemini_client import analyze_text, analyze_image, extract_text_from_image
from app.services.embedder import session_index
from app.services.rag import query_documents
from app.utils.history import clear_history

import logging
logging.basicConfig(level=logging.INFO)

settings = get_settings()
router = APIRouter()


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def analyze_document(
    file: UploadFile | None = File(None),
    text: str | None = Form(None),
    session_id: str = Form("default"),
):
    """Analyze a medical document (PDF, image, or raw text) and return structured insights."""

    # Reset session state
    session_index.reset()
    clear_history(session_id)

    try:
        # ── Determine input ──────────────────────────────────────
        if file is not None:
            content_type = file.content_type or ""
            input_type = detect_input_type(content_type)
            file_bytes = await file.read()

            if not file_bytes:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")

            if input_type == InputType.PDF:
                extracted_text = extract_text_from_pdf(file_bytes)
                if not extracted_text:
                    raise HTTPException(
                        status_code=400,
                        detail="Could not extract text from PDF. The file may be image-only.",
                    )
                insights_dict = await analyze_text(extracted_text)
                index_text = extracted_text

            elif input_type == InputType.IMAGE:
                # Send image directly to Gemini Vision for insights
                insights_dict = await analyze_image(file_bytes, content_type)
                # Also extract raw text for indexing
                index_text = await extract_text_from_image(file_bytes, content_type)

            else:
                # Treat as text file
                extracted_text = file_bytes.decode("utf-8", errors="replace")
                if not extracted_text.strip():
                    raise HTTPException(status_code=400, detail="Uploaded text file is empty")
                insights_dict = await analyze_text(extracted_text)
                index_text = extracted_text

        elif text is not None and text.strip():
            input_type = InputType.TEXT
            insights_dict = await analyze_text(text)
            index_text = text

        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either a file upload or text in the 'text' field.",
            )

        # ── Index for RAG ────────────────────────────────────────
        chunks_count = await session_index.add_text(index_text)

        # ── Validate and return ──────────────────────────────────
        insights = ClinicalInsights(**insights_dict)

        return AnalyzeResponse(
            input_type=input_type,
            insights=insights,
            chunks_indexed=chunks_count,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def query_knowledge(
    request: QueryRequest,
):
    """Ask a question about previously analyzed documents using RAG."""
    try:
        answer, sources = await query_documents(request.question, request.session_id)
        return QueryResponse(answer=answer, sources_used=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(model=settings.GEMINI_MODEL)
