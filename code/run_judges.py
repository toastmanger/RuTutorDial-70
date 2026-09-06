#!/usr/bin/env python3
"""Run LLM judges over the corpora via any OpenAI-compatible API.

Usage:
  .venv/bin/python scripts/run_judges.py                          # full run, all models
  .venv/bin/python scripts/run_judges.py --models gpt-oss-20b --dialogs 2 --repeats 1  # smoke

- temperature 0, verdict per unit = majority of repeats (aggregation is separate).
- Raw responses appended to pilot/judge_runs/raw_<model>.jsonl (resume-safe:
  existing (pilot_id, criterion, repeat) keys are skipped).
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from criteria_prompts import CRITERIA, build_prompt  # noqa: E402
from make_pilot import render_dialog  # noqa: E402

MODELS = [
    "gpt-5.5",
    "anthropic/claude-opus-4.6",
    "google/gemini-3-flash-preview",
    "qwen3-5-397b-a17b-fp8",
    "gpt-oss-120b",
    "google/gemma-3-27b-it",
    "openai/gpt-oss-20b",      # пара к gpt-oss-120b — градиент размера в одном семействе
    "deepseek/deepseek-v3.2",  # седьмое семейство, сильная открытая MoE
]
    # reasoning models reject explicit temperature
NO_TEMP_MODELS = {"gpt-5.5"}
REPEATS = 3
RUNS_DIR = ROOT / "pilot" / "judge_runs"
TRANSPORT_RETRIES = 4
VALID_VERDICTS = {"pass", "fail", "na"}


def load_env() -> None:
    if not os.environ.get("LLM_BASE_URL") or not os.environ.get("LLM_API_KEY"):
        raise RuntimeError("Set LLM_BASE_URL and LLM_API_KEY (any OpenAI-compatible endpoint)")


def parse_verdict(text: str) -> dict | None:
    """Extract {"verdict": ...} JSON from a model response.

    Модели иногда выдают несколько JSON-объектов подряд («Wait, let me
    reconsider» + пересмотренный вердикт) — берём ПОСЛЕДНИЙ валидный.
    strict=False: Claude кладёт буквальные \n внутрь строк JSON.
    """
    text = re.sub(r"```(?:json)?", "", text or "").strip().strip("`")
    dec = json.JSONDecoder(strict=False)
    objs = []
    i = text.find("{")
    while i != -1:
        try:
            obj, end = dec.raw_decode(text, i)
            objs.append(obj)
            i = text.find("{", end)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
    for obj in reversed(objs):
        if isinstance(obj, dict):
            v = str(obj.get("verdict", "")).lower().strip()
            if v in VALID_VERDICTS:
                return {"verdict": v,
                        "evidence": str(obj.get("evidence", ""))[:2000],
                        "reasoning": str(obj.get("reasoning", ""))[:2000]}
    return None


def done_keys(model: str, runs_dir: Path = None) -> set[tuple[str, str, int]]:
    path = (runs_dir or RUNS_DIR) / f"raw_{model.replace('/', '_')}.jsonl"
    keys = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("error"):
                keys.add((r["pilot_id"], r["criterion"], r["repeat"]))
    return keys


async def call_once(client, model: str, prompt: str) -> tuple[str, dict]:
    from llm_client import normalize_model_id
    kwargs = {} if model in NO_TEMP_MODELS else {"temperature": 0}
    resp = await client.chat.completions.create(
        model=normalize_model_id(model),
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    usage = {}
    if getattr(resp, "usage", None):
        usage = {"prompt": resp.usage.prompt_tokens, "completion": resp.usage.completion_tokens}
    return (resp.choices[0].message.content or ""), usage


async def run_unit(client, sem_global, sem_model, model, unit, out_path, stats):
    pilot_id, criterion, repeat, prompt = unit
    async with sem_global, sem_model:
        t0 = time.monotonic()
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model,
                  "pilot_id": pilot_id, "criterion": criterion, "repeat": repeat}
        text, usage, err = "", {}, None
        for attempt in range(TRANSPORT_RETRIES):
            try:
                text, usage = await call_once(client, model, prompt)
                break
            except Exception as e:  # noqa: BLE001 — network/HTTP errors
                err = f"{type(e).__name__}: {e}"
                if attempt < TRANSPORT_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt + random.random())
        parsed = parse_verdict(text) if not err else None
        if not err and parsed is None:
            # по протоколу: один повторный запрос при невалидном JSON
            try:
                text2, usage2 = await call_once(client, model, prompt)
                parsed = parse_verdict(text2)
                if parsed is not None:
                    text, usage = text2, usage2
            except Exception as e:  # noqa: BLE001
                err = f"reask {type(e).__name__}: {e}"
        record["latency_s"] = round(time.monotonic() - t0, 2)
        record["usage"] = usage
        if err and not parsed:
            record["error"] = err
        elif parsed is None:
            record["error"] = "parse_error"
            record["raw_text"] = (text or "")[:3000]
        else:
            record.update(parsed)
        with open(out_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        stats["done"] += 1
        stats["errors"] += 1 if record.get("error") else 0
        if stats["done"] % 50 == 0 or stats["done"] == stats["total"]:
            print(f"[{time.strftime('%H:%M:%S')}] {stats['done']}/{stats['total']} "
                  f"(errors: {stats['errors']})", flush=True)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--dialogs", type=int, default=0, help="limit to first N dialogs (0 = all)")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--concurrency", type=int, default=int(os.environ.get("JUDGE_CONCURRENCY", "16")))
    ap.add_argument("--lang", choices=["en", "ru", "native", "flawed"], default="en",
                    help="ru: перевод (pilot/ru/ru_sample.csv); native: синт-корпус "
                         "(pilot/ru/native_sample.csv); инструкции судьи всегда EN")
    args = ap.parse_args()

    load_env()
    from llm_client import make_client

    cfg = {
        "en": ("pilot_sample.csv", "judge_runs", ("Teacher:", "Student:")),
        "ru": ("ru/ru_sample.csv", "judge_runs_ru", ("Учитель:", "Ученик:")),
        "native": ("ru/native_sample.csv", "judge_runs_native", ("Учитель:", "Ученик:")),
        "flawed": ("ru/native_flawed_sample.csv", "judge_runs_native_flawed", ("Учитель:", "Ученик:")),
    }[args.lang]
    sample_path = ROOT / "pilot" / cfg[0]
    runs_dir = ROOT / "pilot" / cfg[1]
    labels = cfg[2]

    sample = list(csv.DictReader(open(sample_path)))
    sample.sort(key=lambda r: r["pilot_id"])
    if args.dialogs:
        sample = sample[: args.dialogs]

    runs_dir.mkdir(exist_ok=True)
    sem_global = asyncio.Semaphore(args.concurrency)
    tasks = []
    stats = {"done": 0, "errors": 0, "total": 0}
    for model in args.models:
        client = make_client(model)
        sem_model = asyncio.Semaphore(6)
        out_path = runs_dir / f"raw_{model.replace('/', '_')}.jsonl"
        skip = done_keys(model, runs_dir)
        for r in sample:
            dialogue = (render_dialog(r["conversation"], strip_tags=True)
                        .replace("**Учитель:**", labels[0])
                        .replace("**Ученик:**", labels[1]))
            for criterion in CRITERIA:
                prompt = build_prompt(r["question"], r["ground_truth"],
                                      r["student_incorrect_solution"], dialogue, criterion)
                for rep in range(1, args.repeats + 1):
                    if (r["pilot_id"], criterion, rep) in skip:
                        continue
                    unit = (r["pilot_id"], criterion, rep, prompt)
                    tasks.append(run_unit(client, sem_global, sem_model, model, unit, out_path, stats))
    stats["total"] = len(tasks)
    print(f"scheduled: {len(tasks)} calls, models={args.models}, "
          f"concurrency={args.concurrency}", flush=True)
    await asyncio.gather(*tasks)
    print(f"finished: {stats['done']} calls, errors: {stats['errors']}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
