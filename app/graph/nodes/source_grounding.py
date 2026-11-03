import logging

from pydantic import BaseModel

from app.core.dependencies import get_llm_structured
from app.core.source_labels import get_source_label
from app.graph.state import RAGState

logger = logging.getLogger(__name__)

_PROMPT = """\
Analiza la respuesta generada y determina qué documentos del contexto legal fueron \
realmente utilizados para construirla.

RESPUESTA GENERADA:
{answer}

DOCUMENTOS DE CONTEXTO:
{docs_block}

Para cada documento cuyo contenido fue utilizado en la respuesta proporciona:
- Su número de índice entre corchetes
- Un título amigable y descriptivo (máximo 80 caracteres)
- Un resumen de 1 a 2 oraciones sobre el aporte concreto de ese documento

Si ningún documento fue utilizado directamente, devuelve la lista vacía.
"""


class _UsedDocument(BaseModel):
    index: int
    title: str
    summary: str


class _GroundingOutput(BaseModel):
    used: list[_UsedDocument]


async def source_grounding_node(state: RAGState) -> dict:
    docs = state.get("reranked_docs") or []
    answer = state.get("answer", "")
    from_context = bool(state.get("from_context"))

    # Pass-through: no docs to ground (context-only or no retrieval)
    if from_context or not docs or not answer:
        return {
            "sources": [],
            "answer": answer,
            "from_context": from_context,
            "out_of_domain": False,
            "no_results": False,
        }

    docs_block_parts = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.get("metadata", {})
        title = doc.get("title", "Sin título")
        source_type = get_source_label(
            doc.get("source_type", ""), meta.get("resolution_type")
        )
        preview = doc["text"][:400].replace("\n", " ")
        docs_block_parts.append(
            f"[{i}] Tipo: {source_type} | Título: {title}\n    {preview}"
        )
    docs_block = "\n\n".join(docs_block_parts)

    # Salida estructurada (como relevance_filter/context_check): antes se
    # parseaba JSON a mano y 7 de 174 respuestas de la evaluación caían al
    # fallback por JSON inválido.
    used_entries: list[_UsedDocument] = []
    try:
        llm = get_llm_structured(_GroundingOutput)
        result: _GroundingOutput = await llm.ainvoke(
            [{"role": "user", "content": _PROMPT.format(answer=answer, docs_block=docs_block)}]
        )
        used_entries = result.used
    except Exception:
        logger.warning("source_grounding LLM call failed — using all reranked docs as fallback")

    sources = []
    for entry in used_entries:
        if entry.index < 1 or entry.index > len(docs):
            continue
        doc = docs[entry.index - 1]
        meta = doc.get("metadata", {})
        sources.append({
            "source_type": get_source_label(
                doc.get("source_type", ""), meta.get("resolution_type")
            ),
            "source_label": entry.title or doc.get("title", ""),
            "excerpt": entry.summary or doc["text"][:250],
            "score": doc.get("score", 0.0),
        })

    # Fallback: LLM returned nothing valid → expose all reranked docs unchanged
    if not sources:
        for doc in docs:
            meta = doc.get("metadata", {})
            sources.append({
                "source_type": get_source_label(
                    doc.get("source_type", ""), meta.get("resolution_type")
                ),
                "source_label": doc.get("title", ""),
                "excerpt": doc["text"][:250],
                "score": doc.get("score", 0.0),
            })

    return {
        "sources": sources,
        "answer": answer,
        "from_context": False,
        "out_of_domain": False,
        "no_results": False,
    }
