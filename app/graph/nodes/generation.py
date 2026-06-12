from app.core.dependencies import get_llm
from app.core.prompt_provider import prompts
from app.core.source_labels import get_source_label
from app.graph.state import RAGState


def _build_context(docs: list[dict]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.get("metadata", {})
        resolution = meta.get("resolution_reference") or doc.get("source_type", "Documento")
        title = doc.get("title", "")
        source_label = f"{resolution} — {title}".strip(" —") if title else resolution
        parts.append(f"[Documento {i}] ({source_label}):\n{doc['text']}")
    return "\n\n".join(parts)


async def generate_node(state: RAGState) -> dict:
    llm = get_llm()
    messages_history = state.get("messages", [])

    if state.get("context_sufficient"):
        chat_messages = [{"role": "system", "content": prompts.generation}]
        chat_messages += [
            {"role": m["role"], "content": m["content"]} for m in messages_history
        ]
        result = await llm.ainvoke(chat_messages)
        return {
            "answer": result.content.strip(),
            "sources": [],
            "from_context": True,
            "messages": [{"role": "assistant", "content": result.content.strip()}],
        }
    else:
        docs = state.get("reranked_docs") or []
        context_text = _build_context(docs)

        user_prompt = (
            f"Contexto legal recuperado:\n\n{context_text}\n\n"
            f"Pregunta: {state['user_query']}"
        )

        # Include full conversation history for multi-turn coherence
        chat_messages = [{"role": "system", "content": prompts.generation}]
        for m in messages_history[:-1]:  # all turns except the current one
            chat_messages.append({"role": m["role"], "content": m["content"]})
        chat_messages.append({"role": "user", "content": user_prompt})

    result = await llm.ainvoke(chat_messages)
    answer = result.content.strip()

    docs_used = state.get("reranked_docs") or []
    sources = []
    for doc in docs_used:
        meta = doc.get("metadata", {})
        sources.append(
            {
                "source_type": get_source_label(
                    doc.get("source_type", ""),
                    meta.get("resolution_type"),
                ),
                "source_label": doc.get("title", ""),
                "excerpt": doc["text"][:250],
                "score": doc.get("rerank_score", 0.0),
            }
        )

    return {
        "answer": answer,
        "sources": sources,
        "from_context": False,
        "messages": [{"role": "assistant", "content": answer}],
    }
