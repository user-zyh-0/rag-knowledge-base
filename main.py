from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from stage1_min import add_document, list_documents, delete_document, ask, ask_stream
from fastapi.responses import StreamingResponse

app = FastAPI(title="RAG 知识库")


class DocCreate(BaseModel):
    filename: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=255)
    k: int = Field(3, ge=1, le=10)


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
    answer = ask(payload.question, payload.k)
    return {"question": payload.question, "answer": answer}


@app.post("/chat/stream")
def chat_stream(payload: ChatRequest):
    return StreamingResponse(
        ask_stream(payload.question, payload.k),
        media_type="text/plain; charset=utf-8"
    )



if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
