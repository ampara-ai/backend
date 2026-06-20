import random

import httpx

from app.core.config import settings

_MOCK_DIM = 1024


def _is_mock() -> bool:
    return settings.model_service_url.lower() == "mock"


async def get_embedding(text: str, use_base: bool = False) -> list[float]:
    if _is_mock():
        return [random.uniform(-1, 1) for _ in range(_MOCK_DIM)]

    endpoint = settings.embedding_base_endpoint if use_base else settings.embedding_endpoint
    url = settings.model_service_url.rstrip("/") + endpoint
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(url, json={"text": text})
        response.raise_for_status()
        return response.json()["embedding"]


async def rerank(query: str, documents: list[str]) -> list[dict]:
    if _is_mock():
        return [
            {"index": i, "score": 0.9 - i * 0.05, "text": doc}
            for i, doc in enumerate(documents)
        ]

    url = settings.model_service_url.rstrip("/") + settings.reranker_endpoint
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(url, json={"query": query, "documents": documents})
        response.raise_for_status()
        results = response.json()["results"]

    return sorted(results, key=lambda x: x["score"], reverse=True)
