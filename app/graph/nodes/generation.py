from app.core.dependencies import get_llm
from app.core.prompt_provider import prompts
from app.core.source_labels import get_source_label
from app.graph.state import RAGState


def _build_context(docs: list[dict]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.get("metadata", {})
        source_type = doc.get("source_type", "")
        source_name = get_source_label(source_type, meta.get("resolution_type"))
        resolution = "" if source_type == "tipo_01_ley_30364" else meta.get("resolution_reference", "")
        title = doc.get("title", "")
        source_label = " — ".join(part for part in (source_name, resolution, title) if part)
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

    return {
        "answer": answer,
        "sources": [],   # source_grounding_node builds the final sources
        "from_context": False,
        "messages": [{"role": "assistant", "content": answer}],
    }
