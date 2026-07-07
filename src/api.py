"""FastAPI service: POST /ask, GET /health."""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src import config
from src.pipeline import ask

app = FastAPI(title="docqa-eval", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    k: int = Field(default=config.TOP_K, ge=1, le=20)
    mode: str = Field(default=config.RETRIEVAL_MODE, pattern="^(vector|bm25|hybrid)$")
    rerank: bool = Field(default=config.RERANK)
    prompt: str = Field(default=config.PROMPT_VARIANT, pattern="^(naive|grounded)$")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    return ask(req.question, k=req.k, mode=req.mode, rerank=req.rerank,
               prompt_variant=req.prompt)
