import asyncio
import json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.v1.schemas import ChatRequest, ChatResponse, SourceDocument
from app.core.config import settings
from app.graph.rag_graph import clear_session_history, rag_graph
from app.services import ragas_eval

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


def _build_done_event(session_id: str, terminal: dict, answer: str) -> dict:
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
    }


async def _stream_chat_sse(request: ChatRequest) -> AsyncGenerator[str, None]:
    config = {"configurable": {"thread_id": request.session_id}}
    terminal_update: dict = {}
    accumulated_answer: list[str] = []
    generate_status_sent = False

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

        # RAGAS background evaluation — solo para consultas con retrieval real
        if (
            settings.postgres_dsn
            and not terminal_update.get("out_of_domain")
            and not terminal_update.get("no_results")
            and not terminal_update.get("from_context")
            and accumulated_answer  # hubo tokens = generate corrió
        ):
            try:
                final_state = await rag_graph.aget_state(config)
                reranked_docs = final_state.values.get("reranked_docs") or []
                contexts = [doc["text"] for doc in reranked_docs]
                asyncio.create_task(ragas_eval.evaluate_and_save(
                    session_id=request.session_id,
                    user_query=request.message,
                    retrieved_contexts=contexts,
                    answer="".join(accumulated_answer),
                ))
            except Exception:
                logger.exception("Failed to schedule RAGAS task for session %s", request.session_id)

        yield _sse("done", _build_done_event(
            request.session_id, terminal_update, "".join(accumulated_answer)
        ))

    except Exception:
        logger.exception("Stream error for session %s", request.session_id)
        yield _sse("error", {"detail": "Error interno al procesar la consulta."})


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    config = {"configurable": {"thread_id": request.session_id}}
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
        raise HTTPException(status_code=500, detail="Error interno al procesar la consulta.") from exc

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

    # RAGAS background evaluation — solo para consultas con retrieval real
    if (
        settings.ragas_enabled
        and settings.postgres_dsn
        and not result.get("out_of_domain")
        and not result.get("no_results")
        and not result.get("from_context")
    ):
        try:
            final_state = await rag_graph.aget_state(config)
            reranked_docs = final_state.values.get("reranked_docs") or []
            contexts = [doc["text"] for doc in reranked_docs]
            asyncio.create_task(ragas_eval.evaluate_and_save(
                session_id=request.session_id,
                user_query=request.message,
                retrieved_contexts=contexts,
                answer=result.get("answer", ""),
            ))
        except Exception:
            logger.exception("Failed to schedule RAGAS task for session %s", request.session_id)

    return ChatResponse(
        session_id=request.session_id,
        answer=result.get("answer", ""),
        sources=sources,
        from_context=result.get("from_context", False),
        out_of_domain=result.get("out_of_domain", False),
        no_results=result.get("no_results", False),
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
