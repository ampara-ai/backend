import logging

import asyncpg

from app.core.config import settings

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None

# ---------------------------------------------------------------------------
# DDL — created in dependency order so FK references resolve
# ---------------------------------------------------------------------------
_DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS categorias (
        id SERIAL PRIMARY KEY,
        nombre VARCHAR(200) NOT NULL,
        descripcion VARCHAR(500),
        tipo_violencia VARCHAR(100)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS fuentes_legales (
        id SERIAL PRIMARY KEY,
        nombre VARCHAR(200) NOT NULL,
        tipo VARCHAR(100),
        descripcion VARCHAR(500),
        referencia_legal VARCHAR(200)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS admin (
        id SERIAL PRIMARY KEY,
        username VARCHAR(100) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT NOW(),
        is_active BOOLEAN DEFAULT TRUE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS sesiones (
        id UUID PRIMARY KEY,
        fecha DATE DEFAULT CURRENT_DATE,
        duracion_segundos INTEGER DEFAULT 0,
        total_consultas INTEGER DEFAULT 0,
        tuvo_rechazo BOOLEAN DEFAULT FALSE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS consultas (
        id SERIAL PRIMARY KEY,
        sesion_id UUID REFERENCES sesiones(id),
        categoria_id INTEGER REFERENCES categorias(id),
        fuente_citada_id INTEGER REFERENCES fuentes_legales(id),
        fue_relevante BOOLEAN,
        contexto_suficiente BOOLEAN,
        latencia_ms INTEGER,
        chunks_recuperados INTEGER,
        faithfulness FLOAT,
        answer_relevancy FLOAT,
        context_precision FLOAT,
        context_recall FLOAT,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """,
]

_SEED_CATEGORIAS = [
    ("Violencia Física", "Actos que dañan la integridad corporal", "fisica"),
    ("Violencia Psicológica", "Actos que dañan la salud mental", "psicologica"),
    ("Violencia Sexual", "Actos de naturaleza sexual no consentidos", "sexual"),
    ("Violencia Económica o Patrimonial", "Control o privación de recursos económicos", "economica"),
]

_SEED_FUENTES = [
    ("Ley 30364", "ley", "Ley para prevenir y erradicar la violencia familiar", "tipo_01_ley_30364"),
    ("Código Penal", "codigo", "Código Penal del Perú", "tipo_02_codigo_penal"),
    ("Código Procesal Penal", "codigo", "Código Procesal Penal del Perú", "tipo_03_codigo_procesal_penal"),
    ("Jurisprudencia", "jurisprudencia", "Casaciones, acuerdos plenarios y resoluciones", "tipo_04_jurisprudencia"),
    ("Documento Institucional", "institucional", "Protocolos, guías y manuales institucionales", "tipo_05_institucional"),
]


# ---------------------------------------------------------------------------
# Pool management
# ---------------------------------------------------------------------------

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
        for stmt in _DDL_STATEMENTS:
            await conn.execute(stmt)
        for nombre, desc, tipo in _SEED_CATEGORIAS:
            await conn.execute(
                """
                INSERT INTO categorias (nombre, descripcion, tipo_violencia)
                VALUES ($1, $2, $3)
                ON CONFLICT DO NOTHING
                """,
                nombre, desc, tipo,
            )
        for nombre, tipo, desc, ref in _SEED_FUENTES:
            await conn.execute(
                """
                INSERT INTO fuentes_legales (nombre, tipo, descripcion, referencia_legal)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT DO NOTHING
                """,
                nombre, tipo, desc, ref,
            )
        await conn.execute(
            "ALTER TABLE consultas ADD COLUMN IF NOT EXISTS hubo_error BOOLEAN DEFAULT FALSE"
        )
    logger.info("PostgreSQL tables ready")


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


# ---------------------------------------------------------------------------
# Admin user helpers
# ---------------------------------------------------------------------------

async def count_admins() -> int:
    pool = await _get_pool()
    if pool is None:
        return 0
    async with pool.acquire() as conn:
        return await conn.fetchval("SELECT COUNT(*) FROM admin")


async def get_admin_by_username(username: str) -> dict | None:
    pool = await _get_pool()
    if pool is None:
        return None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, username, password_hash, is_active FROM admin WHERE username = $1",
            username,
        )
    return dict(row) if row else None


async def create_admin(username: str, password_hash: str) -> dict:
    pool = await _get_pool()
    if pool is None:
        raise RuntimeError("PostgreSQL not configured")
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO admin (username, password_hash)
            VALUES ($1, $2)
            RETURNING id, username, created_at
            """,
            username,
            password_hash,
        )
    return dict(row)


# ---------------------------------------------------------------------------
# Session + per-query metrics
# ---------------------------------------------------------------------------

async def upsert_session(session_id: str) -> None:
    pool = await _get_pool()
    if pool is None:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO sesiones (id) VALUES ($1) ON CONFLICT (id) DO NOTHING",
                session_id,
            )
    except Exception:
        logger.exception("Failed to upsert session %s", session_id)


async def save_consulta(
    session_id: str,
    fue_relevante: bool,
    contexto_suficiente: bool,
    latencia_ms: int,
    chunks_recuperados: int,
    tuvo_rechazo: bool,
    fuente_citada_id: int | None,
    hubo_error: bool = False,
) -> int | None:
    pool = await _get_pool()
    if pool is None:
        return None
    try:
        async with pool.acquire() as conn:
            consulta_id = await conn.fetchval(
                """
                INSERT INTO consultas
                    (sesion_id, fuente_citada_id, fue_relevante, contexto_suficiente,
                     latencia_ms, chunks_recuperados, hubo_error)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
                """,
                session_id,
                fuente_citada_id,
                fue_relevante,
                contexto_suficiente,
                latencia_ms,
                chunks_recuperados,
                hubo_error,
            )
            await conn.execute(
                """
                UPDATE sesiones
                SET total_consultas = total_consultas + 1,
                    tuvo_rechazo = CASE WHEN $2 THEN TRUE ELSE tuvo_rechazo END
                WHERE id = $1
                """,
                session_id,
                tuvo_rechazo,
            )
        return consulta_id
    except Exception:
        logger.exception("Failed to save consulta for session %s", session_id)
        return None


async def update_consulta_ragas(consulta_id: int, metrics: dict) -> None:
    pool = await _get_pool()
    if pool is None:
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE consultas
                SET faithfulness = $2,
                    answer_relevancy = $3,
                    context_precision = $4,
                    context_recall = $5
                WHERE id = $1
                """,
                consulta_id,
                metrics.get("faithfulness"),
                metrics.get("answer_relevancy"),
                metrics.get("context_precision"),
                metrics.get("context_recall"),
            )
    except Exception:
        logger.exception("Failed to update RAGAS metrics for consulta %d", consulta_id)


async def get_dashboard_metrics() -> dict:
    pool = await _get_pool()
    if pool is None:
        return {"total_queries": 0, "avg_latency_ms": 0.0, "error_rate_pct": 0.0}
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    COUNT(*)                                                                AS total_queries,
                    COALESCE(AVG(latencia_ms) FILTER (WHERE NOT hubo_error), 0)            AS avg_latency_ms,
                    COALESCE(
                        COUNT(*) FILTER (WHERE hubo_error) * 100.0 / NULLIF(COUNT(*), 0),
                        0.0
                    )                                                                       AS error_rate_pct
                FROM consultas
                """
            )
        return {
            "total_queries": row["total_queries"],
            "avg_latency_ms": float(row["avg_latency_ms"]),
            "error_rate_pct": float(row["error_rate_pct"]),
        }
    except Exception:
        logger.exception("Failed to fetch dashboard metrics")
        return {"total_queries": 0, "avg_latency_ms": 0.0, "error_rate_pct": 0.0}


async def get_fuente_id_by_source_type(source_type: str) -> int | None:
    pool = await _get_pool()
    if pool is None:
        return None
    try:
        ref = source_type
        if "jurisprudencia" in source_type.lower() or "tipo_04" in source_type:
            ref = "tipo_04_jurisprudencia"
        elif "institucional" in source_type.lower() or "tipo_05" in source_type:
            ref = "tipo_05_institucional"
        async with pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT id FROM fuentes_legales WHERE referencia_legal = $1", ref
            )
    except Exception:
        logger.exception("Failed to get fuente_id for source_type %s", source_type)
        return None
