"""
Verifica que OpenRouter y MongoDB Atlas respondan correctamente.

Uso:
    uv run python scripts/check_connections.py
"""

import sys
import time

# Permite importar app.* desde la raíz del proyecto
sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])

from app.core.config import settings


def check_mongodb() -> bool:
    print("\n[MongoDB Atlas]")
    try:
        from pymongo import MongoClient

        t0 = time.perf_counter()
        client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=10_000)
        info = client.admin.command("ping")
        latency_ms = (time.perf_counter() - t0) * 1000

        server_info = client.server_info()
        version = server_info.get("version", "?")

        db = client[settings.mongodb_db_name]
        collections = db.list_collection_names()

        print(f"  ✓ Ping OK  ({latency_ms:.0f} ms)")
        print(f"  ✓ MongoDB version: {version}")
        print(f"  ✓ DB: {settings.mongodb_db_name}")
        print(f"  ✓ Colecciones existentes: {collections or '(ninguna aún)'}")

        chunks_col = db[settings.mongodb_collection_chunks]
        count = chunks_col.count_documents({})
        print(f"  ✓ Colección '{settings.mongodb_collection_chunks}': {count} documentos")

        client.close()
        return True
    except Exception as exc:
        print(f"  ✗ Error: {exc}")
        return False


def check_openrouter() -> bool:
    print("\n[OpenRouter LLM]")
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            default_headers={
                "HTTP-Referer": "https://ampara.app",
                "X-Title": "Ampara Legal Chatbot",
            },
        )

        t0 = time.perf_counter()
        response = llm.invoke("Responde solo con la palabra: OK")
        latency_ms = (time.perf_counter() - t0) * 1000

        content = response.content.strip()
        usage = getattr(response, "usage_metadata", None)
        tokens = f", tokens: {usage}" if usage else ""

        print(f"  ✓ Respuesta: '{content}'  ({latency_ms:.0f} ms{tokens})")
        print(f"  ✓ Modelo: {settings.llm_model}")
        return True
    except Exception as exc:
        print(f"  ✗ Error: {exc}")
        return False


def check_model_service() -> bool:
    print("\n[Model Service (ONNX)]")
    if settings.model_service_url.lower() == "mock":
        print("  ~ Modo mock activo — sin verificación de red")
        return True
    try:
        import httpx

        t0 = time.perf_counter()
        r = httpx.get(f"{settings.model_service_url}/health", timeout=5)
        latency_ms = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        print(f"  ✓ Health OK  ({latency_ms:.0f} ms)  → {r.json()}")
        return True
    except Exception as exc:
        print(f"  ✗ Error: {exc}")
        return False


def main():
    print("=" * 50)
    print("  Ampara — Verificación de conexiones")
    print("=" * 50)

    results = {
        "MongoDB Atlas": check_mongodb(),
        "OpenRouter LLM": check_openrouter(),
        "Model Service": check_model_service(),
    }

    print("\n" + "=" * 50)
    print("  Resumen")
    print("=" * 50)
    all_ok = True
    for name, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")
        if not ok:
            all_ok = False

    print()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
