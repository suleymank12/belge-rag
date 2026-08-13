from contextlib import asynccontextmanager

import httpx
import psycopg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.db import get_conn, init_schema

EMBED_TIMEOUT = 30.0   # saniye
CHAT_TIMEOUT = 120.0   # saniye; soğuk model yüklemesi uzun sürebilir

NOT_FOUND_ANSWER = "Bu bilgi yüklü belgelerde bulunmuyor."

SYSTEM_PROMPT = (
    "Yalnızca verilen bağlamdan cevap ver. Bağlamda yoksa 'bu bilgi belgelerde yok' de. "
    "Cevabın sonunda kullandığın kaynakları listele. "
    "Cevabın TAMAMEN Türkçe olmalı; başka bir dilde (İngilizce, Çince vb.) "
    "tek bir kelime bile kullanma."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_schema()
    yield


app = FastAPI(title="belge-rag", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    file: str
    page: int | None
    similarity: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


def _ollama_url(path: str) -> str:
    return settings.ollama_base_url.rstrip("/") + path


def _embed_question(question: str) -> str:
    """Soruyu embed'leyip pgvector literali ('[0.1,...]') döndürür."""
    try:
        resp = httpx.post(
            _ollama_url("/api/embed"),
            json={"model": settings.embed_model, "input": [question]},
            timeout=EMBED_TIMEOUT,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama embedding hatası ({exc.response.status_code}): {exc.response.text.strip()}",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"Ollama'ya erişilemedi: {exc}") from exc
    embeddings = resp.json().get("embeddings") or []
    if not embeddings:
        raise HTTPException(status_code=502, detail="Ollama embedding döndürmedi.")
    return "[" + ",".join(str(x) for x in embeddings[0]) + "]"


def _retrieve(vector: str, top_k: int) -> list[tuple]:
    # <=> cosine DISTANCE döndürür; similarity = 1 - distance
    query = (
        "SELECT source, page, content,"
        "       1 - (embedding <=> %(v)s::vector) AS similarity"
        "  FROM chunks"
        " ORDER BY embedding <=> %(v)s::vector"
        " LIMIT %(k)s"
    )
    try:
        with get_conn() as conn:
            return conn.execute(query, {"v": vector, "k": top_k}).fetchall()
    except psycopg.OperationalError as exc:
        raise HTTPException(status_code=503, detail=f"Veritabanına bağlanılamadı: {exc}") from exc


def _generate_answer(context: str, question: str) -> str:
    payload = {
        "model": settings.chat_model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Bağlam:\n\n{context}\n\nSoru: {question}"},
        ],
    }
    try:
        resp = httpx.post(
            _ollama_url("/v1/chat/completions"), json=payload, timeout=CHAT_TIMEOUT
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama chat hatası ({exc.response.status_code}): {exc.response.text.strip()}",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"Ollama'ya erişilemedi: {exc}") from exc
    try:
        return resp.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise HTTPException(
            status_code=502, detail="Ollama chat cevabı beklenen formatta değil."
        ) from exc


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    vector = _embed_question(req.question)
    rows = _retrieve(vector, req.top_k)

    best = rows[0][3] if rows else None
    if best is None or float(best) < settings.sim_threshold:
        return AskResponse(answer=NOT_FOUND_ANSWER, sources=[])

    context = "\n\n".join(
        f"[kaynak: {source}, s.{page}]\n{content}" for source, page, content, _ in rows
    )
    answer = _generate_answer(context, req.question)
    sources = [
        Source(file=source, page=page, similarity=round(float(sim), 4))
        for source, page, _, sim in rows
    ]
    return AskResponse(answer=answer, sources=sources)


@app.get("/health")
def health():
    return {"status": "ok"}
