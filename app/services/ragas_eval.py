"""
Evaluación RAGAS en background para consultas en vivo.

Métricas calculadas (solo disponibles sin ground truth):
  - faithfulness       : ¿La respuesta es fiel a los contextos recuperados?
  - answer_relevancy   : ¿La respuesta es relevante a la pregunta?

context_precision y context_recall quedan NULL (requieren respuestas de referencia).

Se omite silenciosamente si POSTGRES_DSN no está configurado.
"""

import logging

from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from app.core.config import settings
from app.services.postgres import save_ragas_metrics

logger = logging.getLogger(__name__)


def _build_ragas_tools():
    client = AsyncOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
    )
    llm = llm_factory(settings.ragas_llm_model, provider="openai", client=client)
    embeddings = OpenAIEmbeddings(client=client, model=settings.ragas_embeddings_model)
    return llm, embeddings


async def evaluate_and_save(
    session_id: str,
    user_query: str,
    retrieved_contexts: list[str],
    answer: str,
) -> None:
    if not settings.postgres_dsn:
        return

    try:
        llm, embeddings = _build_ragas_tools()

        faithfulness_metric = Faithfulness(llm=llm)
        relevancy_metric = AnswerRelevancy(llm=llm, embeddings=embeddings)

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
        await save_ragas_metrics(metrics, session_id, user_query)
        logger.info(
            "RAGAS saved — session=%s faithfulness=%.3f answer_relevancy=%.3f",
            session_id,
            f_result.value,
            r_result.value,
        )

    except Exception:
        logger.exception("RAGAS evaluation failed for session %s", session_id)
