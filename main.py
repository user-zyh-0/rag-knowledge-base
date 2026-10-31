from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import uuid

from sqlalchemy.dialects.mysql import limit

from stage1_min import (add_document, list_documents, delete_document,
                        ask, ask_stream, ask_chat, ask_stream_chat, get_history)
from fastapi.responses import StreamingResponse

app = FastAPI(title="RAG 知识库")


class DocCreate(BaseModel):
    filename: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=255)
    k: int = Field(5, ge=1, le=10)
    session_id: str | None = Field(None, max_length=36)


@app.post("/documents", status_code=201)
def create_document(payload: DocCreate):
    info, err = add_document(payload.filename, payload.content)
    if err:
        raise HTTPException(status_code=409, detail=err)
    return info


@app.get("/documents")
def get_documents():
    return list_documents()


@app.delete("/documents/{doc_id}")
def remove_document(doc_id: int):
    info, err = delete_document(doc_id)
    if err:
        raise HTTPException(status_code=404, detail=err)
    return info


@app.post("/chat")
def chat(payload: ChatRequest):
    return ask_chat(payload.question, payload.session_id, payload.k)


@app.post("/chat/stream")
def chat_stream(payload: ChatRequest):
    session_id = payload.session_id or str(uuid.uuid4())
    return StreamingResponse(
        ask_stream_chat(payload.question,session_id, payload.k),
        media_type="text/plain; charset=utf-8",
        headers={"X-session-Id": session_id}
    )


@app.get("/sessions/{session_id}/messages")
def session_messages(session_id: str):
    return get_history(session_id, limit=50)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
