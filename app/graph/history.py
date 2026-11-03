from app.core.config import settings


def previous_turns(messages: list[dict]) -> list[dict]:
    """Mensajes anteriores a la pregunta actual (el último de la lista), limitados a los últimos
    `history_max_pairs` pares pregunta-respuesta. Todos los nodos del grafo usan esta misma ventana."""
    return messages[:-1][-2 * settings.history_max_pairs :]
