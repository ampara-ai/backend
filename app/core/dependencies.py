from langchain_openai import ChatOpenAI

from app.core.config import settings


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        default_headers={
            "HTTP-Referer": "https://ampara.app",
            "X-Title": "Ampara Legal Chatbot",
        },
    )


def get_llm_structured(schema):
    return get_llm().with_structured_output(schema)
