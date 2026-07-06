import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from pydantic import BaseModel

from app.core.config import settings
from app.services import auth, document_processor, model_service, postgres
from app.services.mongodb import get_client

logger = logging.getLogger(__name__)
router = APIRouter()

_ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class CreateAdminRequest(BaseModel):
    username: str
    password: str


class AdminResponse(BaseModel):
    id: int
    username: str
    created_at: str


class UploadResponse(BaseModel):
    filename: str
    chunks_created: int
    chunk_ids: list[str]


class DashboardMetrics(BaseModel):
    avg_latency_ms: int
    total_queries: int
    error_rate_pct: float


class DashboardServices(BaseModel):
    rag_api: str
    inference_service: str


class GeneroBreakdown(BaseModel):
    masculino: int
    femenino: int
    otros: int


class CalificacionesStats(BaseModel):
    utiles: int
    no_utiles: int
    sin_calificar: int


class UsuariosMetrics(BaseModel):
    por_genero: GeneroBreakdown
    calificaciones: CalificacionesStats


class DashboardResponse(BaseModel):
    metrics: DashboardMetrics
    services: DashboardServices
    usuarios: UsuariosMetrics


class QueueItem(BaseModel):
    filename: str
    progress_pct: int
    status: str
    message: str


class QueueResponse(BaseModel):
    queue: list[QueueItem]


class SyncResponse(BaseModel):
    success: bool
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest) -> TokenResponse:
    admin = await postgres.get_admin_by_username(body.username)
    if not admin or not admin["is_active"] or not auth.verify_password(body.password, admin["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario o contraseña incorrectos",
        )
    token = auth.create_access_token({"sub": admin["username"]})
    return TokenResponse(access_token=token)


@router.post("/users", response_model=AdminResponse, status_code=status.HTTP_201_CREATED)
async def create_admin_user(
    body: CreateAdminRequest,
    current_admin: dict = Depends(auth.get_current_admin),
) -> AdminResponse:
    existing = await postgres.get_admin_by_username(body.username)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"El usuario '{body.username}' ya existe",
        )
    password_hash = auth.hash_password(body.password)
    row = await postgres.create_admin(body.username, password_hash)
    return AdminResponse(
        id=row["id"],
        username=row["username"],
        created_at=str(row["created_at"]),
    )


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_admin: dict = Depends(auth.get_current_admin),
) -> UploadResponse:
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Solo se aceptan archivos PDF o DOCX",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El archivo está vacío")

    try:
        chunk_ids = await document_processor.process_and_index(
            filename=file.filename or "documento",
            data=data,
            content_type=file.content_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception:
        logger.exception("Error indexing document '%s'", file.filename)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error al indexar el documento")

    return UploadResponse(
        filename=file.filename or "documento",
        chunks_created=len(chunk_ids),
        chunk_ids=chunk_ids,
    )


@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard(
    current_admin: dict = Depends(auth.get_current_admin),
) -> DashboardResponse:
    metrics, user_metrics, inference_status = await asyncio.gather(
        postgres.get_dashboard_metrics(),
        postgres.get_user_metrics(),
        model_service.health_check(),
    )
    pg = user_metrics["por_genero"]
    cal = user_metrics["calificaciones"]
    return DashboardResponse(
        metrics=DashboardMetrics(
            avg_latency_ms=round(metrics["avg_latency_ms"]),
            total_queries=metrics["total_queries"],
            error_rate_pct=round(metrics["error_rate_pct"], 2),
        ),
        services=DashboardServices(
            rag_api="online",
            inference_service=inference_status,
        ),
        usuarios=UsuariosMetrics(
            por_genero=GeneroBreakdown(
                masculino=pg["masculino"],
                femenino=pg["femenino"],
                otros=pg["otros"],
            ),
            calificaciones=CalificacionesStats(
                utiles=cal["utiles"],
                no_utiles=cal["no_utiles"],
                sin_calificar=cal["sin_calificar"],
            ),
        ),
    )


@router.get("/documents/queue", response_model=QueueResponse)
async def get_document_queue(
    current_admin: dict = Depends(auth.get_current_admin),
) -> QueueResponse:
    client = get_client()
    coll = client[settings.mongodb_db_name][settings.mongodb_collection_chunks]
    pipeline = [
        {"$match": {"doc_type": "documento_subido"}},
        {"$group": {
            "_id": "$doc_id",
            "filename": {"$first": "$title"},
            "chunks": {"$sum": 1},
            "uploaded_at": {"$first": "$metadata.uploaded_at"},
        }},
        {"$sort": {"uploaded_at": -1}},
        {"$limit": 50},
    ]
    docs = await coll.aggregate(pipeline).to_list(50)
    return QueueResponse(
        queue=[
            QueueItem(
                filename=d["filename"],
                progress_pct=100,
                status="active",
                message=f"Indexado con éxito — {d['chunks']} fragmentos",
            )
            for d in docs
        ]
    )


@router.post("/knowledge-base/sync", response_model=SyncResponse)
async def sync_knowledge_base(
    current_admin: dict = Depends(auth.get_current_admin),
) -> SyncResponse:
    client = get_client()
    coll = client[settings.mongodb_db_name][settings.mongodb_collection_chunks]
    total = await coll.count_documents({})
    return SyncResponse(
        success=True,
        message=f"Base de conocimientos actualizada — {total} fragmentos disponibles",
    )
