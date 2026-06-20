"""
Ingesta de chunks legales en MongoDB Atlas y creación de índices de búsqueda.

Uso:
    uv run python scripts/ingest_chunks.py                          # ingesta sin embeddings
    uv run python scripts/ingest_chunks.py --with-embeddings        # ingesta + embeddings finetuned (/embed)
    uv run python scripts/ingest_chunks.py --with-base-embeddings   # ingesta + embeddings base (/embed_base)
    uv run python scripts/ingest_chunks.py --create-indexes-only    # solo crea/verifica índices
    uv run python scripts/ingest_chunks.py --jsonl ./otro.jsonl     # ruta custom del JSONL

El script es idempotente: re-correrlo actualiza documentos existentes (upsert por chunk_id).
Usar --with-embeddings para indexar con el modelo finetuned, --with-base-embeddings para el base.
Los dos flags son mutuamente excluyentes — cada ejecución reindexará la DB con un modelo distinto.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymongo import MongoClient, UpdateOne
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from app.core.config import settings

DEFAULT_JSONL = Path(__file__).resolve().parents[1] / "document_chunks_all.jsonl"
BATCH_SIZE = 32


def load_chunks(jsonl_path: Path) -> list[dict]:
    chunks = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def upsert_chunks(collection, chunks: list[dict]) -> None:
    ops = [
        UpdateOne(
            {"chunk_id": doc["chunk_id"]},
            {"$set": doc},
            upsert=True,
        )
        for doc in chunks
    ]
    result = collection.bulk_write(ops, ordered=False)
    inserted = result.upserted_count
    modified = result.modified_count
    print(f"  ✓ Upserted: {inserted} nuevos, {modified} actualizados  (total: {len(chunks)})")


def add_embeddings(collection, chunks: list[dict], endpoint: str) -> None:
    import httpx

    url = settings.model_service_url.rstrip("/") + endpoint
    total = len(chunks)
    ok = 0
    errors = 0

    print(f"\n  Generando embeddings para {total} chunks (batches de {BATCH_SIZE})...")
    print(f"  Endpoint: {url}")

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_ops = []
        for doc in batch:
            try:
                r = httpx.post(url, json={"text": doc["text"]}, timeout=30)
                r.raise_for_status()
                embedding = r.json()["embedding"]
                batch_ops.append(
                    UpdateOne(
                        {"chunk_id": doc["chunk_id"]},
                        {"$set": {"embedding": embedding}},
                    )
                )
                ok += 1
            except Exception as exc:
                errors += 1
                print(f"    ✗ chunk_id={doc['chunk_id']}: {exc}")

        if batch_ops:
            collection.bulk_write(batch_ops, ordered=False)

        done = min(i + BATCH_SIZE, total)
        pct = done / total * 100
        bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
        print(f"\r  [{bar}] {done}/{total} ({pct:.0f}%)  ok={ok} err={errors}", end="", flush=True)

    print()
    print(f"  ✓ Embeddings completados: {ok} OK, {errors} errores")


def _index_exists(collection, name: str) -> bool:
    try:
        existing = list(collection.list_search_indexes())
        return any(idx.get("name") == name for idx in existing)
    except OperationFailure:
        return False


def create_indexes(collection) -> None:
    print("\n  Verificando/creando índices de búsqueda...")

    # --- Vector Search index ---
    if _index_exists(collection, "vector_index"):
        print("  ~ vector_index ya existe — sin cambios")
    else:
        vector_model = SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": 1024,
                        "similarity": "cosine",
                    }
                ]
            },
            name="vector_index",
            type="vectorSearch",
        )
        collection.create_search_index(vector_model)
        print("  ✓ vector_index creado (estado BUILDING en Atlas — tardará ~1 min en quedar READY)")

    # --- Atlas Search BM25 index ---
    if _index_exists(collection, "text_index"):
        print("  ~ text_index ya existe — sin cambios")
    else:
        text_model = SearchIndexModel(
            definition={
                "mappings": {
                    "dynamic": False,
                    "fields": {
                        "text": [{"type": "string", "analyzer": "lucene.spanish"}],
                        "title": [{"type": "string", "analyzer": "lucene.spanish"}],
                        "retrieval_tags": [{"type": "string"}],
                    },
                }
            },
            name="text_index",
            type="search",
        )
        collection.create_search_index(text_model)
        print("  ✓ text_index creado (estado BUILDING en Atlas — tardará ~1 min en quedar READY)")

    print("  ℹ  Verifica el estado en Atlas UI → Search → Indexes hasta que aparezca READY")


def print_index_status(collection) -> None:
    try:
        indexes = list(collection.list_search_indexes())
        if not indexes:
            print("  (no hay índices de búsqueda aún)")
            return
        for idx in indexes:
            name = idx.get("name", "?")
            status = idx.get("status", "?")
            itype = idx.get("type", "?")
            print(f"  • {name}  tipo={itype}  estado={status}")
    except Exception as exc:
        print(f"  No se pudo listar índices: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Ingesta de chunks + índices MongoDB Atlas")
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL, help="Ruta al archivo JSONL")
    parser.add_argument("--create-indexes-only", action="store_true", help="Solo crear/verificar índices")
    emb_group = parser.add_mutually_exclusive_group()
    emb_group.add_argument("--with-embeddings", action="store_true",
        help="Generar embeddings finetuned via /embed")
    emb_group.add_argument("--with-base-embeddings", action="store_true",
        help="Generar embeddings base via /embed_base")
    args = parser.parse_args()

    print("=" * 56)
    print("  Ampara — Ingesta de chunks + Índices Atlas")
    print("=" * 56)
    print(f"  DB:         {settings.mongodb_db_name}")
    print(f"  Colección:  {settings.mongodb_collection_chunks}")

    client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=15_000)
    collection = client[settings.mongodb_db_name][settings.mongodb_collection_chunks]

    # Verify connection
    try:
        client.admin.command("ping")
        print("  ✓ Conexión Atlas OK\n")
    except Exception as exc:
        print(f"  ✗ No se pudo conectar a MongoDB Atlas: {exc}")
        sys.exit(1)

    if args.create_indexes_only:
        create_indexes(collection)
        print("\n  Estado actual de índices:")
        print_index_status(collection)
        client.close()
        return

    # --- Ingest ---
    jsonl_path: Path = args.jsonl
    if not jsonl_path.exists():
        print(f"  ✗ Archivo no encontrado: {jsonl_path}")
        sys.exit(1)

    print(f"  Cargando chunks desde: {jsonl_path}")
    t0 = time.perf_counter()
    chunks = load_chunks(jsonl_path)
    print(f"  ✓ {len(chunks)} chunks cargados")

    upsert_chunks(collection, chunks)

    # --- Embeddings (opcional) ---
    if args.with_embeddings or args.with_base_embeddings:
        if settings.model_service_url.lower() == "mock":
            flag = "--with-embeddings" if args.with_embeddings else "--with-base-embeddings"
            print(f"\n  ✗ {flag} requiere MODEL_SERVICE_URL real (no 'mock')")
            sys.exit(1)
        if args.with_embeddings:
            add_embeddings(collection, chunks, settings.embedding_endpoint)
        else:
            add_embeddings(collection, chunks, settings.embedding_base_endpoint)

    # --- Indexes ---
    create_indexes(collection)

    elapsed = time.perf_counter() - t0
    print(f"\n  Ingesta completada en {elapsed:.1f}s")

    print("\n  Estado actual de índices:")
    print_index_status(collection)

    client.close()


if __name__ == "__main__":
    main()
