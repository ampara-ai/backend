"""Exactitud de las citas normativas de una respuesta frente a los chunks que recibió el generador (MAJOR #6).

Una cita es un artículo mencionado en la respuesta ("art. 122-B del Código Penal", "artículos 15 y 16 de la Ley 30364"),
con su norma cuando se puede inferir del contexto inmediato. Dos niveles de respaldo:
  estricto   el chunk de ese artículo (misma norma, o cualquier norma si la respuesta no la nombra) estaba en el contexto
  laxo       además, cuenta si el texto de algún chunk del contexto menciona ese artículo (p. ej. un resumen de sentencia)

Métricas: precisión de citas (citas respaldadas / citas) y proporción de respuestas con al menos una cita sin respaldo.
Determinista: no usa un LLM.
"""

import re
from dataclasses import dataclass

CODE_PATTERNS = [
    ("cpp", re.compile(r"c[óo]digo\s+procesal\s+penal|\bn?cpp\b", re.IGNORECASE)),
    ("cp", re.compile(r"c[óo]digo\s+penal|\bc\.?\s?p\.?(?=\W)", re.IGNORECASE)),
    ("ley30364", re.compile(r"ley\s*(n[°º.]?\s*)?30\.?364", re.IGNORECASE)),
]
CHUNK_CODE = {"tipo01": "ley30364", "tipo02": "cp", "tipo03": "cpp"}
# número de artículo: 15, 122-B, 122B (la letra va pegada o con guion; "15 y 16" son dos artículos)
_NUM = r"\d{1,3}(?:[-‑]?[a-zA-Z](?![a-zA-Z]))?"
ARTICLE = re.compile(r"\b(?:art[íi]culos?|arts?\.)\s*(?:n[°º.]?\s*)?(" + _NUM + r"(?:\s*(?:,|y|e)\s*" + _NUM + r")*)",
                     re.IGNORECASE)
ARTICLE_NUMBER = re.compile(_NUM)
WINDOW = 80  # caracteres después de la cita en los que se busca la norma ("art. 15 de la Ley 30364")
LEAD = 40    # caracteres antes de la cita, dentro de la misma oración ("el NCPP, art. 95")


def norm_article(raw: str) -> str:
    raw = raw.strip().lower().replace("‑", "-").replace(" ", "-")
    m = re.fullmatch(r"(\d+)-?([a-z])?", raw)
    return f"{m.group(1)}-{m.group(2)}" if m and m.group(2) else (m.group(1) if m else raw)


@dataclass(frozen=True)
class Citation:
    article: str
    code: str | None  # ley30364 / cp / cpp, o None si la respuesta no nombra la norma
    span: str


def extract_citations(text: str) -> list[Citation]:
    out: list[Citation] = []
    for m in ARTICLE.finditer(text):
        after = re.split(r"[.;\n]\s", text[m.end(): m.end() + WINDOW], maxsplit=1)[0]
        before = text[max(0, m.start() - WINDOW): m.start()]
        code = next((c for c, p in CODE_PATTERNS if p.search(after)), None)
        if code is None:  # "(Ley 30364, art. 15)": la norma va antes del artículo, en el mismo paréntesis
            paren = before[before.rfind("(") + 1:] if "(" in before and ")" not in before[before.rfind("("):] else ""
            code = next((c for c, p in CODE_PATTERNS if p.search(paren)), None)
        if code is None:  # "el NCPP, art. 95": la norma justo antes, en la misma oración
            lead = re.split(r"[.;\n]\s", before[-LEAD:])[-1]
            code = next((c for c, p in CODE_PATTERNS if p.search(lead)), None)
        for num in ARTICLE_NUMBER.findall(m.group(1)):
            out.append(Citation(norm_article(num), code, m.group(0)))
    return list(dict.fromkeys(out))


def chunk_article(chunk_id: str) -> tuple[str, str] | None:
    """chunk_tipo02_cp_art_122-b_01 -> ("cp", "122-b"); jurisprudencia e institucionales -> None."""
    m = re.match(r"chunk_(tipo0[123])_[a-z0-9]+_art_([0-9]+(?:-[a-z])?)_\d+$", chunk_id)
    return (CHUNK_CODE[m.group(1)], m.group(2)) if m else None


def supported(cit: Citation, context_ids: list[str], context_texts: list[str]) -> tuple[bool, bool]:
    arts = {a for a in map(chunk_article, context_ids) if a}
    strict = any(art == cit.article and (cit.code is None or code == cit.code) for code, art in arts)
    if strict:
        return True, True
    pattern = re.compile(r"\bart[íi]culos?\s*(?:n[°º.]?\s*)?" + re.escape(cit.article.split("-")[0])
                         + (r"[-‑ ]?" + cit.article.split("-")[1] if "-" in cit.article else r"(?![-‑ ]?[a-z]\b)") + r"\b",
                         re.IGNORECASE)
    return False, any(pattern.search(t) for t in context_texts)


def score_answer(answer: str, context_ids: list[str], context_texts: list[str]) -> dict:
    cits = extract_citations(answer)
    verdicts = [supported(c, context_ids, context_texts) for c in cits]
    return {
        "n_citations": len(cits),
        "supported_strict": sum(s for s, _ in verdicts),
        "supported_lenient": sum(l for _, l in verdicts),
        "unsupported": [f"{c.code or '?'}:{c.article}" for c, (_, l) in zip(cits, verdicts) if not l],
    }


def aggregate(rows: list[dict]) -> dict:
    n = sum(r["n_citations"] for r in rows)
    with_cit = [r for r in rows if r["n_citations"]]
    return {
        "answers": len(rows), "answers_with_citations": len(with_cit), "citations": n,
        "precision_strict": sum(r["supported_strict"] for r in rows) / n if n else None,
        "precision_lenient": sum(r["supported_lenient"] for r in rows) / n if n else None,
        "answers_with_unsupported_citation": sum(bool(r["unsupported"]) for r in with_cit) / len(with_cit) if with_cit else None,
    }
