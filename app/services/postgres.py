import json
import logging

import asyncpg

from app.core.config import settings

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ragas_metrics (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    session_id TEXT,
    query TEXT,
    faithfulness FLOAT,
    answer_relevancy FLOAT,
    context_precision FLOAT,
    context_recall FLOAT,
    raw JSONB
);
"""


async def _get_pool() -> asyncpg.Pool | None:
    global _pool
    if not settings.postgres_dsn:
        return None
    if _pool is None:
        _pool = await asyncpg.create_pool(settings.postgres_dsn)
    return _pool


async def create_tables_if_not_exist() -> None:
    pool = await _get_pool()
    if pool is None:
        return
    async with pool.acquire() as conn:
        await conn.execute(_CREATE_TABLE_SQL)


async def save_ragas_metrics(metrics: dict, session_id: str, query: str) -> None:
    pool = await _get_pool()
    if pool is None:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ragas_metrics
                    (session_id, query, faithfulness, answer_relevancy,
                     context_precision, context_recall, raw)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                session_id,
                query,
                metrics.get("faithfulness"),
                metrics.get("answer_relevancy"),
                metrics.get("context_precision"),
                metrics.get("context_recall"),
                json.dumps(metrics),
            )
    except Exception:
        logger.exception("Failed to save RAGAS metrics")


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
