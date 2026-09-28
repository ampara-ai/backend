import json
import logging

from app.core.dependencies import get_llm
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
- Un título amigable y descriptivo (máximo 80 caracteres)
- Un resumen de 1 a 2 oraciones sobre el aporte concreto de ese documento

Responde ÚNICAMENTE con JSON válido en este formato exacto:
{{"used": [{{"index": 1, "title": "...", "summary": "..."}}, ...]}}

Si ningún documento fue utilizado directamente responde: {{"used": []}}
"""


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

    used_entries: list[dict] = []
    try:
        llm = get_llm()
        result = await llm.ainvoke(
            [{"role": "user", "content": _PROMPT.format(answer=answer, docs_block=docs_block)}]
        )
        raw = result.content.strip()
        # Strip markdown code fences if the model wraps the JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        used_entries = json.loads(raw).get("used", [])
    except Exception:
        logger.warning("source_grounding LLM call failed — using all reranked docs as fallback")

    sources = []
    for entry in used_entries:
        idx = entry.get("index")
        if not isinstance(idx, int) or idx < 1 or idx > len(docs):
            continue
        doc = docs[idx - 1]
        meta = doc.get("metadata", {})
        sources.append({
            "source_type": get_source_label(
                doc.get("source_type", ""), meta.get("resolution_type")
            ),
            "source_label": entry.get("title") or doc.get("title", ""),
            "excerpt": entry.get("summary") or doc["text"][:250],
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
