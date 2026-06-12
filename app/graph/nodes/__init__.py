def out_of_domain_node(state: dict) -> dict:
    return {
        "answer": (
            "Lo siento, solo puedo responder preguntas sobre violencia familiar "
            "y protección legal en el Perú (Ley 30364 y normativa relacionada)."
        ),
        "out_of_domain": True,
        "sources": [],
    }


def no_results_node(state: dict) -> dict:
    return {
        "answer": (
            "No encontré información suficientemente relevante en mi base de conocimiento "
            "para responder esta pregunta con certeza. Te recomiendo consultar directamente "
            "a una institución especializada como el CEM (Centro de Emergencia Mujer) "
            "o la Fiscalía de Familia."
        ),
        "no_results": True,
        "sources": [],
    }
