"""
Evaluación RAGAS en background para consultas en vivo.

Métricas calculadas (solo disponibles sin ground truth):
  - faithfulness       : ¿La respuesta es fiel a los contextos recuperados?
  - answer_relevancy   : ¿La respuesta es relevante a la pregunta?

context_precision y context_recall quedan NULL (requieren respuestas de referencia).

Se omite silenciosamente si POSTGRES_DSN no está configurado o ragas no está disponible.
"""

import logging

from app.core.config import settings
from app.services.postgres import update_consulta_ragas

logger = logging.getLogger(__name__)


def _build_ragas_tools():
    # Lazy imports — ragas has version-specific transitive deps that may not
    # be present in all environments. Keeping them inside the function lets
    # the server start even if ragas is partially broken.
    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, Faithfulness

    client = AsyncOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
    )
    llm = llm_factory(settings.ragas_llm_model, provider="openai", client=client,max_tokens=10000)
    embeddings = OpenAIEmbeddings(client=client, model=settings.ragas_realtime_embeddings_model)
    faithfulness_metric = Faithfulness(llm=llm)
    relevancy_metric = AnswerRelevancy(llm=llm, embeddings=embeddings)
    return faithfulness_metric, relevancy_metric


async def evaluate_and_save(
    consulta_id: int,
    user_query: str,
    retrieved_contexts: list[str],
    answer: str,
) -> None:
    if not settings.postgres_dsn:
        return

    try:
        faithfulness_metric, relevancy_metric = _build_ragas_tools()

        f_result = await faithfulness_metric.ascore(
            user_input=user_query,
            response=answer,
            retrieved_contexts=retrieved_contexts,
        )
        r_result = await relevancy_metric.ascore(
            user_input=user_query,
            response=answer,
        )

        metrics = {
            "faithfulness": f_result.value,
            "answer_relevancy": r_result.value,
            "context_precision": None,
            "context_recall": None,
        }
        await update_consulta_ragas(consulta_id, metrics)
        logger.info(
            "RAGAS saved — consulta_id=%d faithfulness=%.3f answer_relevancy=%.3f",
            consulta_id,
            f_result.value,
            r_result.value,
        )

    except Exception:
        logger.exception("RAGAS evaluation failed for consulta_id %d", consulta_id)
