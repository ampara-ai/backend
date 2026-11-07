"""Procedencia de una corrida: qué código, qué LLM, qué temperatura y qué prompts produjeron un resultado.

La evaluación de la Tabla 13 no registró el prompt del generador y el archivo cambió después; desde la revisión Q1, cada
corrida que llama al LLM guarda este bloque junto a sus salidas.
"""

import datetime
import hashlib
import subprocess
from pathlib import Path

from app.core.config import settings
from app.core.dependencies import _provider_preferences
from app.core.prompt_provider import _PROMPTS_DIR, prompts

ROOT = Path(__file__).resolve().parents[1]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_state(repo: Path = ROOT) -> dict:
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""
    return {"commit": git("rev-parse", "HEAD") or None, "dirty": bool(git("status", "--porcelain"))}


def llm_provenance() -> dict:
    """Configuración efectiva del LLM del grafo y hash de cada prompt tal como se cargó en memoria."""
    return {
        "llm_model": settings.llm_model,
        "llm_temperature": settings.llm_temperature,
        "llm_max_output_tokens": settings.llm_max_output_tokens,
        "provider": _provider_preferences(),
        "generation_prompt_file": prompts.generation_file,
        "prompts_sha256": {
            "generation": sha256_text(prompts.generation),
            "relevance_filter": sha256_text(prompts.relevance_filter),
            "context_check": sha256_text(prompts.context_check),
        },
        "prompts_dir": str(_PROMPTS_DIR),
    }


def run_provenance(**extra) -> dict:
    return {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git": git_state(),
        "llm": llm_provenance(),
        **extra,
    }
