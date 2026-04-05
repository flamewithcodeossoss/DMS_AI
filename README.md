# Medical Document Management System Analyzer

An AI-powered medical document analysis system that extracts structured clinical insights from PDFs, images, and raw text using a Retrieval-Augmented Generation (RAG) pipeline. Both the backend **and** the frontend are fully hosted on **Modal** (serverless) — no local server needed.

**Live App:** https://waitforossoss14--medical-rag-dms-streamlit-ui.modal.run
**API Docs:** https://waitforossoss14--medical-rag-dms-fastapi-app.modal.run/docs

---

## Table of Contents

1. [Approach](#approach)
2. [Models Used](#models-used)
3. [Prompt Design Strategy](#prompt-design-strategy)
4. [Example I/O](#example-io)
5. [How to Run the Project](#how-to-run-the-project)

---

## Approach

The system follows a pipeline adapted for medical document understanding:

### 1. Document Ingestion & Extraction
- **PDFs** → text extracted page-by-page using `pymupdf`
- **Images** (PNG, JPEG, GIF, WebP) → sent directly to a vision-capable LLM to extract text and insights simultaneously
- **Raw text** → accepted and processed directly

### 2. Chunking & Embedding
Extracted text is split into overlapping chunks (default: 1000 characters with 200-character overlap) to preserve context across boundaries. Each chunk is then embedded using a dense text embedding model via the OpenRouter API.

### 3. Vector Indexing (FAISS)
Chunk embeddings are stored in an **in-memory FAISS `IndexFlatL2`** index per session. This allows fast nearest-neighbour retrieval without a persistent database — keeping the architecture simple and stateless on the server side.

### 4. Clinical Insight Extraction
On document upload, the full document text (or image) is sent to the LLM with a structured clinical prompt. The model returns a **strict JSON object** containing patient information, symptoms, diagnosis, lab/imaging results, medications, next steps, and a plain-English human summary.

### Architecture Overview

```
User (Streamlit UI)
       │
       ▼
  POST /analyze          POST /query
       │                      │
       ▼                      ▼
  FastAPI (Modal)    ──────────────────
       │             Retrieve top-K chunks
       │             from FAISS index
       ▼                      │
  LLM via OpenRouter  ◄───────┘
  (structured JSON output / RAG answer)
```

---

## Models Used

| Role | Model | Provider |
|---|---|---|
| **LLM (analysis & Q&A)** | `qwen/qwen3.6-plus:free` | OpenRouter → Qwen |
| **Vision (image docs)** | `qwen/qwen3.6-plus:free` (multimodal) | OpenRouter → Qwen |
| **Embeddings** | `nvidia/llama-nemotron-embed-vl-1b-v2:free` | OpenRouter → NVIDIA |

All models are accessed via the **OpenRouter API** using an OpenAI-compatible client (`openai` Python SDK pointed at `https://openrouter.ai/api/v1`). No local GPU is required.

---

## Prompt Design Strategy

### Document Analysis Prompt (Structured Extraction)

The system uses a **strict schema-enforcement** strategy. The LLM is given a detailed system prompt that:

1. **Defines a fixed JSON schema** the model must return — no markdown, no code fences, only raw JSON.
2. **Lists explicit rules** to prevent hallucination (`"If a field is not found, set it to null"`).
3. **Requires plain-language output** (`human_summary`) alongside technical fields, so both clinicians and patients can read it.

```
You are an expert medical document analyst.
Your job is to extract structured clinical insights from medical text.

You MUST return ONLY valid JSON with this exact schema:
{
  "patient_info": { "name", "age", "gender", "id", "date_of_report", "additional" },
  "symptoms": [ { "name", "severity", "duration", "notes" } ],
  "medical_assessment": { "diagnosis", "findings", "lab_results", "imaging_results",
                          "medications", "notes" },
  "suggested_next_steps": [ { "action", "priority", "reason" } ],
  "human_summary": "3-5 sentence plain-English summary"
}

Rules:
- Extract ALL available information. Set missing fields to null.
- Be precise with medical terminology.
- Do NOT hallucinate information not present in the source text.
- Return ONLY the JSON object, no extra text.
```

## Example I/O

### Example 1 — PDF Analysis

**Input:** Upload a discharge summary PDF

**Output:**
```json
{
  "patient_info": {
    "name": "John Doe",
    "age": "54",
    "gender": "Male",
    "id": "MRN-00123",
    "date_of_report": "2026-03-28",
    "additional": {}
  },
  "symptoms": [
    { "name": "Chest pain", "severity": "Moderate", "duration": "3 days", "notes": "Radiating to left arm" },
    { "name": "Shortness of breath", "severity": "Mild", "duration": "2 days", "notes": null }
  ],
  "medical_assessment": {
    "diagnosis": "Acute Myocardial Infarction (NSTEMI)",
    "findings": ["ST depression in V4-V6", "Elevated troponin T"],
    "lab_results": { "troponin_T": "0.9 ng/mL", "CK-MB": "elevated" },
    "imaging_results": "Echocardiogram: EF 45%, mild hypokinesia of anterior wall",
    "medications": ["Aspirin 100mg daily", "Atorvastatin 40mg nightly", "Metoprolol 25mg twice daily"],
    "notes": "Patient admitted via emergency; responded well to dual antiplatelet therapy."
  },
  "suggested_next_steps": [
    { "action": "Cardiology follow-up in 2 weeks", "priority": "high", "reason": "Monitor EF recovery post-NSTEMI" },
    { "action": "Repeat ECG and troponin at 6 hours", "priority": "high", "reason": "Rule out further ischaemia" }
  ],
  "human_summary": "John Doe, a 54-year-old male, was admitted with chest pain and mild breathlessness and diagnosed with a type of heart attack called NSTEMI. Blood tests and heart imaging confirmed the diagnosis. He was started on blood thinners and cholesterol medication and responded well. He should follow up with a cardiologist within two weeks to monitor his heart function."
}
```

---

### Example 3 — Image Upload (Lab Report)

**Input:** A JPEG image of a blood test report

**Output:**
```json
{
  "patient_info": { "name": "Jane Smith", "age": "34", "gender": "Female", ... },
  "symptoms": [],
  "medical_assessment": {
    "diagnosis": null,
    "findings": ["Haemoglobin: 9.2 g/dL (low)", "MCV: 71 fL (low)", "Ferritin: 6 ng/mL (low)"],
    "lab_results": { "haemoglobin": "9.2 g/dL", "MCV": "71 fL", "ferritin": "6 ng/mL" },
    "imaging_results": null,
    "medications": [],
    "notes": "Results consistent with iron-deficiency anaemia."
  },
  "suggested_next_steps": [
    { "action": "Iron supplementation", "priority": "high", "reason": "Low ferritin indicates depleted iron stores" }
  ],
  "human_summary": "Jane Smith's blood test shows she has low haemoglobin and iron levels, which is consistent with iron-deficiency anaemia. Her red blood cells appear smaller than normal. It is recommended to start iron supplements and follow up with a doctor to investigate the underlying cause of the iron deficiency."
}
```

---

## How to Run the Project

### Option A — Use the Live Hosted App (no setup needed)

| Service | URL |
|---|---|
| **Streamlit UI** | https://waitforossoss14--medical-rag-dms-streamlit-ui.modal.run |
| **FastAPI backend** | https://waitforossoss14--medical-rag-dms-fastapi-app.modal.run |
| **API docs (Swagger)** | https://waitforossoss14--medical-rag-dms-fastapi-app.modal.run/docs |

Just open the Streamlit URL in your browser and start uploading documents.

---

### Option B — Redeploy to Modal yourself

#### Prerequisites

- Python 3.12+
- [Modal account](https://modal.com) with CLI installed
- OpenRouter API key (free tier is sufficient)

#### 1. Clone & Install Dependencies

```bash
git clone <repo-url>
cd DMS-RAG

python -m venv dmspo
dmspo\Scripts\activate        # Windows
# source dmspo/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

#### 2. Configure Modal Secret

```bash
# First-time Modal login
modal setup

# Create the secret (replace with your real key)
modal secret create medical-rag-secrets \
  GEMINI_API_KEY=your_openrouter_api_key_here \
  MODAL_API_URL=https://<your-username>--medical-rag-dms-fastapi-app.modal.run
```

> `GEMINI_API_KEY` holds an **OpenRouter** key. Get yours at https://openrouter.ai/keys

#### 3. Deploy Both Frontend and Backend

```bash
modal deploy modal_app.py
```

Modal will print two URLs after deployment:

```
https://<your-username>--medical-rag-dms-fastapi-app.modal.run   ← backend
https://<your-username>--medical-rag-dms-streamlit-ui.modal.run  ← frontend (open this)
```

Open the **`streamlit-ui`** URL in your browser — done.

---

### Option C — Local Development (no Modal)

```bash
# Terminal 1 — backend
uvicorn app.main:app --reload --port 8000

# Terminal 2 — frontend
MODAL_API_URL=http://localhost:8000 streamlit run streamlit_app.py
```

---

### API Endpoints (reference)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/analyze` | Analyze a medical document (PDF / image / text) |
| `POST` | `/query` | Ask a follow-up question using RAG |
| `POST` | `/query` | Ask a follow-up question using RAG |
