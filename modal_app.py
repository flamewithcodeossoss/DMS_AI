import modal

# ── Backend image (FastAPI) ──────────────────────────────────
backend_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi[standard]",
        "openai",
        "sentence-transformers",
        "pymupdf",
        "faiss-cpu",
        "numpy",
        "pydantic-settings",
        "python-multipart",
    )
    .add_local_dir("app", remote_path="/root/app")
)

# ── Frontend image (Streamlit) ───────────────────────────────
frontend_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "streamlit",
        "requests",
    )
    .add_local_file("streamlit_app.py", remote_path="/root/streamlit_app.py")
)

app = modal.App(name="medical-rag-dms")


# ── Backend endpoint ─────────────────────────────────────────
@app.function(
    image=backend_image,
    secrets=[modal.Secret.from_name("medical-rag-secrets")],
    timeout=300,
    memory=1024,
)
@modal.asgi_app()
def fastapi_app():
    from app.main import app as web_app
    return web_app


# ── Frontend endpoint ────────────────────────────────────────
@app.function(
    image=frontend_image,
    secrets=[modal.Secret.from_name("medical-rag-secrets")],
    timeout=3600,
    memory=512,
    max_containers=1,
    min_containers=1,
)
@modal.concurrent(max_inputs=100)
@modal.web_server(8501)
def streamlit_ui():
    import subprocess
    subprocess.Popen(
        [
            "streamlit", "run", "/root/streamlit_app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true",
            "--server.enableCORS=false",
            "--server.enableXsrfProtection=false",
            "--server.maxUploadSize=100",
            "--server.enableWebsocketCompression=false",
            "--browser.gatherUsageStats=false",
        ]
    )
