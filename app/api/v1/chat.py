import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api.v1.schemas import ChatRequest, ChatResponse, SourceDocument
from app.core.config import settings
from app.graph.rag_graph import clear_session_history, rag_graph
from app.services import postgres, ragas_eval

logger = logging.getLogger(__name__)
router = APIRouter()

_NODE_STATUS: dict[str, str | None] = {
    "relevance_filter": "Verificando relevancia...",
    "context_check": "Analizando contexto...",
    "query_synthesis": "Reformulando consulta...",
    "embed_query": "Generando embedding...",
    "bm25_search": "Buscando (léxico)...",
    "semantic_search": "Buscando (semántico)...",
    "rrf_fusion": "Fusionando resultados...",
    "rerank": "Reordenando resultados...",
    # "generate" status is emitted on first token (before "updates" arrives)
    "out_of_domain": None,
    "no_results": None,
}

_TERMINAL_NODES = {"generate", "out_of_domain", "no_results"}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _build_done_event(session_id: str, terminal: dict, answer: str, consulta_id: int | None = None) -> dict:
    raw_sources = terminal.get("sources") or []
    sources = [
        {
            "source_type": s.get("source_type", ""),
            "source_label": s.get("source_label", ""),
            "excerpt": s.get("excerpt", ""),
            "score": float(s.get("score", 0.0)),
        }
        for s in raw_sources
    ]
    return {
        "session_id": session_id,
        "answer": terminal.get("answer", answer),
        "sources": sources,
        "from_context": bool(terminal.get("from_context", False)),
        "out_of_domain": bool(terminal.get("out_of_domain", False)),
        "no_results": bool(terminal.get("no_results", False)),
        "consulta_id": consulta_id,
    }


async def _save_metrics(
    session_id: str,
    terminal: dict,
    latencia_ms: int,
    reranked_docs: list[dict],
    answer: str,
    user_query: str,
    hubo_error: bool = False,
) -> int | None:
    """Persist session + consulta rows; schedule RAGAS if enabled. Returns consulta_id."""
    if not settings.postgres_dsn:
        return None

    out_of_domain = bool(terminal.get("out_of_domain"))
    no_results = bool(terminal.get("no_results"))
    from_context = bool(terminal.get("from_context"))
    fue_relevante = not out_of_domain
    tuvo_rechazo = out_of_domain or no_results
    chunks_recuperados = len(reranked_docs)

    fuente_citada_id: int | None = None
    raw_sources = terminal.get("sources") or []
    if raw_sources:
        top_source_type = raw_sources[0].get("source_type", "")
        fuente_citada_id = await postgres.get_fuente_id_by_source_type(top_source_type)

    await postgres.upsert_session(session_id)
    consulta_id = await postgres.save_consulta(
        session_id=session_id,
        fue_relevante=fue_relevante,
        contexto_suficiente=from_context,
        latencia_ms=latencia_ms,
        chunks_recuperados=chunks_recuperados,
        tuvo_rechazo=tuvo_rechazo,
        fuente_citada_id=fuente_citada_id,
        hubo_error=hubo_error,
    )

    if (
        settings.ragas_enabled
        and consulta_id is not None
        and not out_of_domain
        and not no_results
        and not from_context
        and reranked_docs
        and answer
    ):
        contexts = [doc["text"] for doc in reranked_docs]
        asyncio.create_task(ragas_eval.evaluate_and_save(
            consulta_id=consulta_id,
            user_query=user_query,
            retrieved_contexts=contexts,
            answer=answer,
        ))

    return consulta_id


async def _stream_chat_sse(request: ChatRequest) -> AsyncGenerator[str, None]:
    config = {"configurable": {"thread_id": request.session_id}}
    terminal_update: dict = {}
    accumulated_answer: list[str] = []
    generate_status_sent = False
    start_time = time.time()

    try:
        async for mode, data in rag_graph.astream(
            {
                "session_id": request.session_id,
                "user_query": request.message,
                "messages": [{"role": "user", "content": request.message}],
            },
            config=config,
            stream_mode=["updates", "messages"],
        ):
            if mode == "updates":
                for node_name, update in data.items():
                    status_msg = _NODE_STATUS.get(node_name)
                    if status_msg:
                        yield _sse("status", {"step": node_name, "message": status_msg})
                    if node_name in _TERMINAL_NODES:
                        terminal_update = update

            elif mode == "messages":
                token, metadata = data
                if (
                    metadata.get("langgraph_node") == "generate"
                    and hasattr(token, "content")
                    and token.content
                ):
                    if not generate_status_sent:
                        yield _sse("status", {"step": "generate", "message": "Generando respuesta..."})
                        generate_status_sent = True
                    accumulated_answer.append(token.content)
                    yield _sse("token", {"content": token.content})

        latencia_ms = int((time.time() - start_time) * 1000)
        answer = "".join(accumulated_answer)

        try:
            final_state = await rag_graph.aget_state(config)
            reranked_docs = final_state.values.get("reranked_docs") or []
        except Exception:
            reranked_docs = []

        consulta_id = await _save_metrics(
            session_id=request.session_id,
            terminal=terminal_update,
            latencia_ms=latencia_ms,
            reranked_docs=reranked_docs,
            answer=answer,
            user_query=request.message,
        )

        yield _sse("done", _build_done_event(request.session_id, terminal_update, answer, consulta_id))

    except Exception:
        logger.exception("Stream error for session %s", request.session_id)
        asyncio.create_task(_save_metrics(
            session_id=request.session_id,
            terminal={},
            latencia_ms=int((time.time() - start_time) * 1000),
            reranked_docs=[],
            answer="",
            user_query=request.message,
            hubo_error=True,
        ))
        yield _sse("error", {"detail": "Error interno al procesar la consulta."})


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    config = {"configurable": {"thread_id": request.session_id}}
    start_time = time.time()
    try:
        result = await rag_graph.ainvoke(
            {
                "session_id": request.session_id,
                "user_query": request.message,
                "messages": [{"role": "user", "content": request.message}],
            },
            config=config,
        )
    except Exception as exc:
        logger.exception("RAG graph error for session %s", request.session_id)
        asyncio.create_task(_save_metrics(
            session_id=request.session_id,
            terminal={},
            latencia_ms=int((time.time() - start_time) * 1000),
            reranked_docs=[],
            answer="",
            user_query=request.message,
            hubo_error=True,
        ))
        raise HTTPException(status_code=500, detail="Error interno al procesar la consulta.") from exc

    latencia_ms = int((time.time() - start_time) * 1000)

    raw_sources = result.get("sources") or []
    sources = [
        SourceDocument(
            source_type=s.get("source_type", ""),
            source_label=s.get("source_label", ""),
            excerpt=s.get("excerpt", ""),
            score=float(s.get("score", 0.0)),
        )
        for s in raw_sources
    ]

    try:
        final_state = await rag_graph.aget_state(config)
        reranked_docs = final_state.values.get("reranked_docs") or []
    except Exception:
        reranked_docs = []

    consulta_id = await _save_metrics(
        session_id=request.session_id,
        terminal=result,
        latencia_ms=latencia_ms,
        reranked_docs=reranked_docs,
        answer=result.get("answer", ""),
        user_query=request.message,
    )

    return ChatResponse(
        session_id=request.session_id,
        answer=result.get("answer", ""),
        sources=sources,
        from_context=result.get("from_context", False),
        out_of_domain=result.get("out_of_domain", False),
        no_results=result.get("no_results", False),
        consulta_id=consulta_id,
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream_chat_sse(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete("/session/{session_id}")
async def clear_session(session_id: str) -> dict:
    clear_session_history(session_id)
    return {"cleared": True, "session_id": session_id}


class GeneroRequest(BaseModel):
    genero: Literal["masculino", "femenino", "otros"]


@router.post("/session/{session_id}/genero")
async def set_session_genero(session_id: str, body: GeneroRequest) -> dict:
    await postgres.update_session_genero(session_id, body.genero)
    return {"ok": True}


class RatingRequest(BaseModel):
    session_id: str
    fue_util: bool


@router.post("/consulta/{consulta_id}/rate")
async def rate_consulta(consulta_id: int, body: RatingRequest) -> dict:
    updated = await postgres.update_consulta_rating(consulta_id, body.session_id, body.fue_util)
    if not updated:
        raise HTTPException(status_code=404, detail="Consulta no encontrada para esta sesión")
    return {"ok": True}
