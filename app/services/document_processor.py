import io
import logging
import re
from datetime import datetime, timezone

import tiktoken
from pymongo import UpdateOne

from app.services import model_service
from app.services.mongodb import get_client
from app.core.config import settings

logger = logging.getLogger(__name__)

CHUNK_SIZE = 500
OVERLAP = 50


def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")[:60]


def extract_text_pdf(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n".join(parts)


def extract_text_docx(data: bytes) -> str:
    from docx import Document as DocxDocument
    doc = DocxDocument(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def chunk_text(text: str) -> list[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += CHUNK_SIZE - OVERLAP
    return chunks


async def process_and_index(filename: str, data: bytes, content_type: str) -> list[str]:
    if content_type == "application/pdf":
        text = extract_text_pdf(data)
    else:
        text = extract_text_docx(data)

    if not text.strip():
        raise ValueError("No se pudo extraer texto del documento")

    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("El documento no generó ningún chunk")

    slug = _slugify(filename.rsplit(".", 1)[0])
    uploaded_at = datetime.now(timezone.utc).isoformat()
    chunk_ids: list[str] = []
    operations: list[UpdateOne] = []

    for i, chunk_text_str in enumerate(chunks):
        chunk_id = f"upload_{slug}_{i:04d}"
        chunk_ids.append(chunk_id)

        embedding = await model_service.get_embedding(chunk_text_str)

        doc = {
            "chunk_id": chunk_id,
            "doc_id": f"upload_{slug}",
            "doc_type": "documento_subido",
            "source_type": "documento_subido",
            "chunk_type": "simple_chunk",
            "title": filename,
            "text": chunk_text_str,
            "order": i,
            "embedding": embedding,
            "metadata": {
                "original_filename": filename,
                "uploaded_at": uploaded_at,
            },
            "retrieval_tags": ["documento_subido"],
        }
        operations.append(UpdateOne({"chunk_id": chunk_id}, {"$set": doc}, upsert=True))

    client = get_client()
    collection = client[settings.mongodb_db_name][settings.mongodb_collection_chunks]
    result = await collection.bulk_write(operations, ordered=False)
    logger.info(
        "Indexed %d chunks from '%s' (upserted=%d, modified=%d)",
        len(chunks), filename, result.upserted_count, result.modified_count,
    )
    return chunk_ids
