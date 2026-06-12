from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from app.core.config import settings

_client: AsyncIOMotorClient | None = None

_COMMON_PROJECT = {
    "_id": {"$toString": "$_id"},
    "chunk_id": 1,
    "title": 1,
    "text": 1,
    "doc_type": 1,
    "source_type": 1,
    "metadata": 1,
    "retrieval_tags": 1,
}


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.mongodb_uri)
    return _client


def get_chunks_collection() -> AsyncIOMotorCollection:
    return get_client()[settings.mongodb_db_name][settings.mongodb_collection_chunks]


async def vector_search(embedding: list[float], k: int = 20) -> list[dict]:
    collection = get_chunks_collection()
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": k * 10,
                "limit": k,
            }
        },
        {
            "$project": {
                **_COMMON_PROJECT,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    cursor = collection.aggregate(pipeline)
    return await cursor.to_list(length=k)


async def bm25_search(query: str, k: int = 20) -> list[dict]:
    collection = get_chunks_collection()
    pipeline = [
        {
            "$search": {
                "index": "text_index",
                "compound": {
                    "should": [
                        {"text": {"query": query, "path": "text"}},
                        {
                            "text": {
                                "query": query,
                                "path": "title",
                                "score": {"boost": {"value": 1.5}},
                            }
                        },
                        {
                            "text": {
                                "query": query,
                                "path": "retrieval_tags",
                                "score": {"boost": {"value": 1.2}},
                            }
                        },
                    ]
                },
            }
        },
        {"$limit": k},
        {
            "$project": {
                **_COMMON_PROJECT,
                "score": {"$meta": "searchScore"},
            }
        },
    ]
    cursor = collection.aggregate(pipeline)
    return await cursor.to_list(length=k)


async def close_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
