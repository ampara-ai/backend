"""Juicio de relevancia del pooling por API (OpenRouter): resuelve las peticiones pendientes que dejó `pool_eval.py export`.

Cada petición (output/work/pool_test/requests/<task>.json) trae el prompt del sistema y los pares (pregunta, chunk) de unas pocas
preguntas. El script llama al modelo juez, valida que haya un veredicto por cada par y escribe la respuesta en
output/work/pool_test/responses/<task>.json (con la clave `_model`, que `pool_eval.py ingest` registra como procedencia).
Es reanudable: solo toma las peticiones sin respuesta, así que se puede repetir tras un corte o un error de la API.

Variables: OPENROUTER_API_KEY y OPENROUTER_BASE_URL (las lee app/core/config.py desde .env), POOL_JUDGE_MODEL (id de OpenRouter del
modelo juez; por defecto openai/gpt-6.1-sol).

Uso (desde la raíz del repo):
    uv run python quantitative_experiments/pooling/run_judge.py [--concurrency 6] [--effort medium] [--limit N]
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from openai import AsyncOpenAI

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from app.core.config import settings  # noqa: E402

OUT = Path(os.environ.get("POOL_OUT", HERE / "output"))
REQ_DIR, RES_DIR = OUT / "work" / "pool_test" / "requests", OUT / "work" / "pool_test" / "responses"


def expected_pairs(user: str) -> set[tuple[str, str]]:
    payload = json.loads(user[user.index("["):])
    return {(q["dataset_id"], c["chunk_id"]) for q in payload for c in q["chunks"]}


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


async def judge_one(client: AsyncOpenAI, model: str, effort: str, req_path: Path, sem: asyncio.Semaphore, retries: int = 4) -> bool:
    req = json.loads(req_path.read_text(encoding="utf-8"))
    want = expected_pairs(req["user"])
    for attempt in range(1, retries + 1):
        try:
            async with sem:
                r = await client.chat.completions.create(
                    model=model, temperature=0,
                    messages=[{"role": "system", "content": req["system"]}, {"role": "user", "content": req["user"]}],
                    response_format={"type": "json_object"}, extra_body={"reasoning": {"effort": effort}})
            resp = parse_json(r.choices[0].message.content or "")
            got = {(j["dataset_id"], j["chunk_id"]) for j in resp.get("judgments") or []}
            if not want <= got or any(j.get("relevance") not in ("si", "parcial", "no") for j in resp["judgments"]):
                raise ValueError(f"respuesta incompleta o inválida ({len(want - got)} pares sin veredicto)")
            resp["_model"] = model
            RES_DIR.mkdir(parents=True, exist_ok=True)
            (RES_DIR / req_path.name).write_text(json.dumps(resp, ensure_ascii=False, indent=1), encoding="utf-8")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"  {req_path.stem}: intento {attempt}/{retries} falló: {exc}")
            await asyncio.sleep(2 * attempt)
    return False


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("POOL_JUDGE_MODEL", "openai/gpt-6.1-sol"))
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="solo las N primeras pendientes (0 = todas)")
    args = ap.parse_args()

    done = {p.name for p in RES_DIR.glob("*.json")} if RES_DIR.exists() else set()
    pending = [p for p in sorted(REQ_DIR.glob("*.json")) if p.name not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"peticiones pendientes: {len(pending)} | juez: {args.model}")
    if not pending:
        return
    client = AsyncOpenAI(base_url=settings.openrouter_base_url, api_key=settings.openrouter_api_key)
    sem = asyncio.Semaphore(args.concurrency)
    ok = await asyncio.gather(*(judge_one(client, args.model, args.effort, p, sem) for p in pending))
    print(f"resueltas: {sum(ok)}/{len(pending)}. Siguiente: python quantitative_experiments/pooling/pool_eval.py ingest")
    sys.exit(0 if all(ok) else 3)


if __name__ == "__main__":
    asyncio.run(main())
