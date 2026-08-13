"""PDF ingest: metni sayfa sayfa çıkar, chunk'la, embed'le, chunks tablosuna yaz.

Kullanım: python -m ingest.ingest dosya.pdf

Aynı dosya tekrar ingest edilirse önce eski kayıtlar silinir (source bazlı).
Silme + ekleme tek transaction'da yapılır; embedding aşaması başarısız olursa
veritabanına hiçbir şey yazılmaz.
"""

import argparse
import sys
from pathlib import Path

import httpx
import psycopg
from pypdf import PdfReader

from app.config import settings
from app.db import get_conn, init_schema

CHUNK_SIZE = 1500      # hedef chunk uzunluğu (karakter)
CHUNK_OVERLAP = 200    # ardışık chunk'lar arası örtüşme (karakter)
EMBED_BATCH_SIZE = 32  # tek /api/embed çağrısına giden chunk sayısı
EMBED_DIM = 1024       # chunks.embedding vector(1024) ile eşleşmeli

# Öncelik sırasıyla bölme noktaları: paragraf, cümle, satır, kelime.
# Ayırıcılar bir önceki parçanın sonunda korunur; kelime ortasından kesilmez.
SEPARATORS = ["\n\n", ". ", "! ", "? ", "\n", " "]


def _split_keep_sep(text: str, sep: str) -> list[str]:
    parts = text.split(sep)
    return [p + sep for p in parts[:-1]] + [parts[-1]]


def _split_recursive(text: str, separators: list[str]) -> list[str]:
    """CHUNK_SIZE'ı aşan metni ayırıcı hiyerarşisine göre küçük parçalara böler."""
    if len(text) <= CHUNK_SIZE or not separators:
        return [text]
    sep, rest = separators[0], separators[1:]
    if sep not in text:
        return _split_recursive(text, rest)
    pieces: list[str] = []
    for piece in _split_keep_sep(text, sep):
        if len(piece) <= CHUNK_SIZE:
            pieces.append(piece)
        else:
            pieces.extend(_split_recursive(piece, rest))
    return pieces


def _overlap_tail(chunk: str) -> str:
    """Chunk sonundan ~CHUNK_OVERLAP karakterlik, kelime sınırında başlayan kuyruk."""
    if len(chunk) <= CHUNK_OVERLAP:
        return chunk
    tail = chunk[-CHUNK_OVERLAP:]
    space = tail.find(" ")
    return tail[space + 1:] if space != -1 else tail


def chunk_text(text: str) -> list[str]:
    pieces = _split_recursive(text, SEPARATORS)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > CHUNK_SIZE:
            chunks.append(current.strip())
            current = _overlap_tail(current)
        current += piece
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]


def extract_pages(path: Path) -> list[tuple[int, str]]:
    """PDF'ten (sayfa_no, metin) listesi döndürür; boş sayfaları atlar."""
    reader = PdfReader(str(path))
    pages = []
    for page_no, page in enumerate(reader.pages, start=1):
        # \x00 Postgres text kolonunda geçersiz; bazı PDF'lerden sızabiliyor
        text = (page.extract_text() or "").replace("\x00", "").strip()
        if text:
            pages.append((page_no, text))
    return pages


def embed_texts(texts: list[str]) -> list[list[float]]:
    url = f"{settings.ollama_base_url.rstrip('/')}/api/embed"
    embeddings: list[list[float]] = []
    with httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            try:
                resp = client.post(url, json={"model": settings.embed_model, "input": batch})
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                sys.exit(
                    f"HATA: Ollama {exc.response.status_code} döndürdü: {exc.response.text.strip()}\n"
                    f"'{settings.embed_model}' modeli yüklü mü? (ollama pull {settings.embed_model})\n"
                    "Veritabanına hiçbir şey yazılmadı."
                )
            except httpx.HTTPError as exc:
                sys.exit(
                    f"HATA: Ollama'ya erişilemedi ({url}): {exc}\n"
                    "Ollama çalışıyor mu? Veritabanına hiçbir şey yazılmadı."
                )
            batch_embeddings = resp.json().get("embeddings") or []
            if len(batch_embeddings) != len(batch):
                sys.exit(
                    f"HATA: Ollama {len(batch)} chunk için {len(batch_embeddings)} embedding döndürdü. "
                    "Veritabanına hiçbir şey yazılmadı."
                )
            embeddings.extend(batch_embeddings)
            print(f"  embedding: {min(i + EMBED_BATCH_SIZE, len(texts))}/{len(texts)}")
    if embeddings and len(embeddings[0]) != EMBED_DIM:
        sys.exit(
            f"HATA: Embedding boyutu {len(embeddings[0])}, şema vector({EMBED_DIM}) bekliyor. "
            f"EMBED_MODEL={settings.embed_model} doğru mu?"
        )
    return embeddings


def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(str(x) for x in embedding) + "]"


def store_chunks(source: str, rows: list[tuple[int, str, list[float]]]) -> None:
    """Eski kayıtları siler ve yenilerini ekler — tek transaction."""
    with get_conn() as conn:
        conn.execute("DELETE FROM chunks WHERE source = %s", (source,))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (source, page, content, embedding)"
                " VALUES (%s, %s, %s, %s::vector)",
                [(source, page, content, _vector_literal(emb)) for page, content, emb in rows],
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PDF'i chunk'layıp bge-m3 ile embed'leyerek chunks tablosuna yazar."
    )
    parser.add_argument("pdf", help="ingest edilecek PDF dosyası")
    args = parser.parse_args()

    path = Path(args.pdf)
    if not path.is_file():
        sys.exit(f"HATA: Dosya bulunamadı: {path}")

    pages = extract_pages(path)
    if not pages:
        sys.exit(f"HATA: PDF'ten metin çıkarılamadı: {path.name}")

    page_chunks = [(page_no, chunk) for page_no, text in pages for chunk in chunk_text(text)]
    print(f"{path.name}: {len(pages)} dolu sayfa, {len(page_chunks)} chunk.")

    # Embedding'e para/zaman harcamadan önce DB erişimini ve şemayı doğrula
    try:
        init_schema()
    except psycopg.OperationalError as exc:
        sys.exit(f"HATA: Veritabanına bağlanılamadı: {exc}\n'docker compose up -d' çalışıyor mu?")

    embeddings = embed_texts([chunk for _, chunk in page_chunks])

    source = path.name
    rows = [(page_no, chunk, emb) for (page_no, chunk), emb in zip(page_chunks, embeddings)]
    try:
        store_chunks(source, rows)
    except psycopg.OperationalError as exc:
        sys.exit(f"HATA: Veritabanına yazılamadı: {exc}\nHiçbir kayıt eklenmedi (transaction geri alındı).")
    print(f"Tamam: {len(rows)} chunk '{source}' kaynağıyla kaydedildi.")


if __name__ == "__main__":
    main()
