import base64
import json
import os
from openai import AsyncOpenAI
from app.core.config import get_settings

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            settings = get_settings()
            api_key = settings.GEMINI_API_KEY.strip()
            
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing! Set it in your .env file or Modal secrets.")
            
        _client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://medical-rag-dms.modal.run",
                "X-OpenRouter-Title": "Medical RAG DMS",
            },
        )
    return _client


CLINICAL_SYSTEM_PROMPT = """You are an expert medical document analyst. 
Your job is to extract structured clinical insights from medical text.

You MUST return ONLY valid JSON (no markdown, no code fences) with this exact schema:
{
  "patient_info": {
    "name": "string or null",
    "age": "string or null",
    "gender": "string or null",
    "id": "string or null",
    "date_of_report": "string or null",
    "additional": {}
  },
  "symptoms": [
    {"name": "string", "severity": "string or null", "duration": "string or null", "notes": "string or null"}
  ],
  "medical_assessment": {
    "diagnosis": "string or null",
    "findings": ["string"],
    "lab_results": {},
    "imaging_results": "string or null",
    "medications": ["string"],
    "notes": "string or null"
  },
  "suggested_next_steps": [
    {"action": "string", "priority": "high|medium|low or null", "reason": "string or null"}
  ],
  "human_summary": "A clear, concise 3-5 sentence summary a doctor or patient can read."
}

Rules:
- Extract ALL available information from the medical text.
- If a field is not found in the text, set it to null or empty list.
- Be precise with medical terminology.
- The human_summary should be in plain English, understandable by a non-medical person.
- Do NOT hallucinate information not present in the source text.
- Return ONLY the JSON object, no extra text."""


async def analyze_text(text: str, context: str = "") -> dict:
    """Send extracted text to Gemini for clinical insight extraction."""
    prompt = f"{CLINICAL_SYSTEM_PROMPT}\n\n"
    if context:
        prompt += f"Previous context:\n{context}\n\n"
    prompt += f"Medical document text:\n{text}"

    settings = get_settings()
    response = await _get_client().chat.completions.create(
        model=settings.GEMINI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        extra_body={},
    )
    return _parse_json_response(response.choices[0].message.content)


async def analyze_image(image_bytes: bytes, mime_type: str, context: str = "") -> dict:
    """Send image directly to Gemini Vision for clinical insight extraction."""
    b64 = base64.b64encode(image_bytes).decode()

    prompt = CLINICAL_SYSTEM_PROMPT
    if context:
        prompt += f"\n\nPrevious context:\n{context}"
    prompt += "\n\nAnalyze the medical document in this image:"

    settings = get_settings()
    response = await _get_client().chat.completions.create(
        model=settings.GEMINI_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
            ],
        }],
        extra_body={},
    )
    return _parse_json_response(response.choices[0].message.content)


async def answer_query(question: str, rag_context: str, history: str = "") -> str:
    """Answer a question using RAG context and conversation history."""
    prompt = (
        "You are a medical document assistant. Answer the question using ONLY "
        "the provided context from previously analyzed medical documents.\n"
        "If the answer is not in the context, say so clearly.\n\n"
    )
    if history:
        prompt += f"Conversation history:\n{history}\n\n"
    prompt += f"Context from documents:\n{rag_context}\n\n"
    prompt += f"Question: {question}"

    settings = get_settings()
    response = await _get_client().chat.completions.create(
        model=settings.GEMINI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        extra_body={},
    )
    return response.choices[0].message.content


async def extract_text_from_image(image_bytes: bytes, mime_type: str) -> str:
    """Use Gemini Vision to extract raw text from a medical document image."""
    b64 = base64.b64encode(image_bytes).decode()
    prompt = (
        "Extract ALL text visible in this medical document image. "
        "Return ONLY the extracted text, preserving the original structure. "
        "Do not summarize or interpret — just extract the raw text."
    )
    settings = get_settings()
    response = await _get_client().chat.completions.create(
        model=settings.GEMINI_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
            ],
        }],
        extra_body={},
    )
    return response.choices[0].message.content


def _parse_json_response(text: str) -> dict:
    """Parse JSON from Gemini response, handling markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]  # remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    return json.loads(cleaned)
