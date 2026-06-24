import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.admin import router as admin_router
from app.api.v1.chat import router as chat_router
from app.services.mongodb import close_client, get_client
from app.services.postgres import close_pool, create_tables_if_not_exist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: verify MongoDB connection
    try:
        client = get_client()
        await client.admin.command("ping")
        logger.info("MongoDB connection OK")
    except Exception:
        logger.warning("MongoDB not reachable at startup — continuing anyway")

    await create_tables_if_not_exist()

    yield

    # Shutdown
    await close_client()
    await close_pool()
    logger.info("MongoDB client closed")


app = FastAPI(
    title="Ampara API",
    description="Backend RAG legal — violencia familiar Perú",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1/admin", tags=["admin"])


@app.get("/health")
async def health():
    return {"status": "ok"}
