import psycopg

from app.config import settings

# Bilinçli karar: embedding kolonuna ANN index (ivfflat/hnsw) eklenmedi.
# Bu veri boyutunda exact (sequential scan) arama yeterli; gerekirse
# sonradan index eklenebilir.
SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id        serial PRIMARY KEY,
    source    text NOT NULL,
    page      int,
    content   text NOT NULL,
    embedding vector(1024)
);
"""


def get_conn() -> psycopg.Connection:
    return psycopg.connect(settings.database_url)


def init_schema() -> None:
    with get_conn() as conn:
        conn.execute(SCHEMA_SQL)
