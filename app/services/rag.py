from app.services.embedder import session_index
from app.services.gemini_client import answer_query
from app.utils.history import get_history, add_message


async def query_documents(question: str, session_id: str) -> tuple[str, int]:
    """Retrieve relevant chunks and answer question using Gemini."""
    chunks = await session_index.search(question)
    if not chunks:
        return "No documents have been indexed in this session. Please upload a document first using /analyze.", 0

    context = "\n\n---\n\n".join(chunks)
    history = get_history(session_id)

    answer = await answer_query(question, context, history)

    add_message(session_id, role="user", content=question)
    add_message(session_id, role="assistant", content=answer)

    return answer, len(chunks)
