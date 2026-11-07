"""Cliente LLM de los experimentos de la revisión (OpenRouter, API compatible con OpenAI).

* Caché en disco (JSONL de solo anexar): la clave es el SHA-256 del modelo, los mensajes y los parámetros, así que repetir una
  corrida no vuelve a pagar llamadas hechas y una corrida cortada se reanuda donde quedó.
* `dry_run=True`: no llama a la API; cuenta las peticiones que se harían y guarda la primera para revisarla. Así se puede probar
  un script completo sin clave ni costo.
* Respuestas JSON (`response_format=json_object`); se reintenta si la respuesta no es JSON válido o si `validate` la rechaza.

No importa app.core.config: los scripts deben poder ejecutarse en seco sin .env.
"""

import asyncio
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv

# La clave vive en el .env del backend (no versionado); las variables ya definidas en el entorno tienen prioridad.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


def parse_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


def backoff(exc: Exception, attempt: int) -> float:
    """Segundos antes del reintento: los límites de tasa (429, p. ej. 20 por minuto en cuentas nuevas) esperan una ventana."""
    rate_limited = getattr(exc, "status_code", None) == 429 or "429" in str(exc)[:200]
    return min(65.0, 15.0 * attempt) if rate_limited else 2.0 * attempt


def request_key(model: str, messages: list[dict], params: dict) -> str:
    blob = json.dumps({"model": model, "messages": messages, "params": params}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class LLM:
    def __init__(self, model: str, cache_path: Path, *, temperature: float = 0.0, max_tokens: int = 8000,
                 concurrency: int = 6, dry_run: bool = False, reasoning_effort: str | None = None,
                 provider: dict | None = None, retries: int = 6) -> None:
        self.model, self.cache_path, self.dry_run, self.retries = model, Path(cache_path), dry_run, retries
        self.params = {"temperature": temperature, "max_tokens": max_tokens}
        self.extra_body: dict = {}
        if reasoning_effort:
            self.extra_body["reasoning"] = {"effort": reasoning_effort}
        if provider:
            self.extra_body["provider"] = provider
        self.sem = asyncio.Semaphore(concurrency)
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["key"]] = row["response"]
        self.pending: list[list[dict]] = []
        self.calls = 0
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("falta OPENROUTER_API_KEY (o usa --dry-run)")
            self._client = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)
        return self._client

    def _store(self, key: str, messages: list[dict], response: dict) -> None:
        self.cache[key] = response
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "model": self.model, "params": self.params, "messages": messages,
                                "response": response}, ensure_ascii=False) + "\n")

    async def json(self, system: str, user: str, validate: Callable[[dict], None] | None = None) -> dict | None:
        """Devuelve el objeto JSON de la respuesta (o None en seco). `validate` lanza una excepción si la respuesta no sirve."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        key = request_key(self.model, messages, {**self.params, **self.extra_body})
        if key in self.cache:
            return self.cache[key]
        if self.dry_run:
            self.pending.append(messages)
            return None
        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                async with self.sem:
                    r = await self.client.chat.completions.create(
                        model=self.model, messages=messages, response_format={"type": "json_object"},
                        extra_body=self.extra_body or None, **self.params)
                self.calls += 1
                resp = parse_json(r.choices[0].message.content)
                if validate:
                    validate(resp)
                self._store(key, messages, resp)
                return resp
            except Exception as exc:  # noqa: BLE001
                last = exc
                await asyncio.sleep(backoff(exc, attempt))
        raise RuntimeError(f"{self.model}: sin respuesta válida tras {self.retries} intentos: {last}")

    async def text(self, messages: list[dict]) -> str | None:
        """Respuesta de texto libre (generación), con la misma caché y el mismo modo en seco."""
        key = request_key(self.model, messages, {**self.params, **self.extra_body, "format": "text"})
        if key in self.cache:
            return self.cache[key]["text"]
        if self.dry_run:
            self.pending.append(messages)
            return None
        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                async with self.sem:
                    r = await self.client.chat.completions.create(model=self.model, messages=messages,
                                                                  extra_body=self.extra_body or None, **self.params)
                self.calls += 1
                content = (r.choices[0].message.content or "").strip()
                if not content:
                    raise ValueError("respuesta vacía")
                self._store(key, messages, {"text": content})
                return content
            except Exception as exc:  # noqa: BLE001
                last = exc
                await asyncio.sleep(backoff(exc, attempt))
        raise RuntimeError(f"{self.model}: sin respuesta tras {self.retries} intentos: {last}")

    def report(self) -> str:
        if self.dry_run:
            first = json.dumps(self.pending[0], ensure_ascii=False, indent=1)[:3000] if self.pending else "(ninguna)"
            return (f"[seco] {self.model}: {len(self.pending)} peticiones nuevas, {len(self.cache)} ya en caché "
                    f"({self.cache_path}).\nPrimera petición:\n{first}")
        return f"{self.model}: {self.calls} llamadas nuevas, caché en {self.cache_path}"
