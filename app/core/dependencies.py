from langchain_openai import ChatOpenAI

from app.core.config import settings


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        # Tope de tokens de salida — sin esto, un mensaje corto tipo "cuenta
        # del 1 al 100000" puede hacer que el LLM genere una respuesta
        # arbitrariamente larga y cara. El filtro de largo de entrada
        # (ChatRequest.message max_length) no alcanza a frenar este caso
        # porque el mensaje en sí es corto.
        max_tokens=settings.llm_max_output_tokens,
        default_headers={
            "HTTP-Referer": "https://ampara.app",
            "X-Title": "Ampara Legal Chatbot",
        },
    )


def get_llm_structured(schema):
    return get_llm().with_structured_output(schema)
