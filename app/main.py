from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Medical Document RAG System",
        description="Transform unstructured medical reports into structured clinical insights using Gemini 2.5 Flash + FAISS RAG",
        version="1.0.0",
        servers=[{"url": "/", "description": "Current Environment"}]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()
