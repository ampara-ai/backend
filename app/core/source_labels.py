_SOURCE_TYPE_MAP: dict[str, str] = {
    "tipo_01_ley_30364": "Ley 30364",
    "tipo_02_codigo_penal": "Código Penal",
    "tipo_03_codigo_procesal_penal": "Código Procesal Penal",
}

_JURISPRUDENCIA_MAP: dict[str, str] = {
    "Casacion": "Jurisprudencia · Casación",
    "Acuerdo Plenario": "Jurisprudencia · Acuerdo Plenario",
    "Expediente / Sentencia de Instancia": "Jurisprudencia · Sentencia",
    "Habeas Corpus / Amparo (TC)": "Jurisprudencia · TC",
    "Recurso de Nulidad": "Jurisprudencia · Recurso de Nulidad",
    "Resolucion Administrativa": "Jurisprudencia · Resolución Administrativa",
}

_INSTITUCIONAL_MAP: dict[str, str] = {
    "coleccion_guias": "Guía Institucional",
    "protocolo": "Protocolo",
    "protocolo_institucional": "Protocolo",
    "servicio_orientacion": "Manual Institucional",
}


def get_source_label(source_type: str, resolution_type: str | None) -> str:
    if source_type in _SOURCE_TYPE_MAP:
        return _SOURCE_TYPE_MAP[source_type]

    if source_type == "tipo_04_jurisprudencia":
        return _JURISPRUDENCIA_MAP.get(resolution_type or "", "Jurisprudencia")

    if source_type == "tipo_05_documentos_institucionales":
        return _INSTITUCIONAL_MAP.get(resolution_type or "", "Documento Institucional")

    return "Documento Legal"
