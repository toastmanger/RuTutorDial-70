#!/usr/bin/env python3
"""Run LLM judges over the corpora via any OpenAI-compatible API.

Usage:
  python code/run_judges.py                          # full run, all models
  python code/run_judges.py --models gpt-oss-20b --dialogs 2 --repeats 1  # smoke

- temperature 0, verdict per unit = majority of repeats (aggregation is separate).
- Raw responses appended to <logs>/<runs_dir>/raw_<model>.jsonl, see layout.py (resume-safe:
  existing (pilot_id, criterion, repeat) keys are skipped).
- Endpoint and key: LLM_BASE_URL, LLM_API_KEY (see llm_client.py).
- --interleave --seed N: all calls (models x corpora x units x repeats) are shuffled
  with a fixed seed, so repeats of one unit and the EN/RU versions are spread over
  the whole run window instead of being sent back-to-back (v0.5 control run).
- Each JSONL record carries ts_start/ts_end (UTC, ms), the model id returned by the
  provider, finish_reason and the request parameters actually sent.
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
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from criteria_prompts import CRITERIA, build_prompt  # noqa: E402
from layout import DATA, LOGS, RU  # noqa: E402
from make_pilot import render_dialog  # noqa: E402

MODELS = [
    "gpt-5.5",
    "anthropic/claude-opus-4.6",
    "google/gemini-3-flash-preview",
    "qwen3-5-397b-a17b-fp8",
    "gpt-oss-120b",
    "google/gemma-3-27b-it",
    "openai/gpt-oss-20b",      # same family as gpt-oss-120b: size gradient
    "deepseek/deepseek-v3.2",
]
# gpt-5.5 rejects an explicit temperature: the parameter is not sent at all
NO_TEMP_MODELS = {"gpt-5.5", "openai/gpt-5.5"}
REPEATS = 3
RUNS_DIR = LOGS / "judge_runs"
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


def request_params(model: str) -> dict:
    """Sampling parameters sent to the API (top_p / max_tokens are never sent)."""
    return {} if model in NO_TEMP_MODELS else {"temperature": 0}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


async def call_once(client, model: str, prompt: str) -> tuple[str, dict, dict]:
    from llm_client import normalize_model_id
    resp = await client.chat.completions.create(
        model=normalize_model_id(model),
        messages=[{"role": "user", "content": prompt}],
        **request_params(model),
    )
    usage = {}
    if getattr(resp, "usage", None):
        usage = {"prompt": resp.usage.prompt_tokens, "completion": resp.usage.completion_tokens}
    meta = {"served_model": getattr(resp, "model", None),
            "system_fingerprint": getattr(resp, "system_fingerprint", None),
            "finish_reason": resp.choices[0].finish_reason if resp.choices else None}
    return (resp.choices[0].message.content or ""), usage, meta


async def run_unit(client, sem_global, sem_model, model, unit, out_path, stats, extra=None):
    pilot_id, criterion, repeat, prompt = unit
    # model semaphore first: a task waiting for its model must not hold a global slot
    async with sem_model, sem_global:
        t0 = time.monotonic()
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "ts_start": utc_now(), "model": model,
                  "pilot_id": pilot_id, "criterion": criterion, "repeat": repeat,
                  "params": request_params(model), **(extra or {})}
        text, usage, meta, err = "", {}, {}, None
        for attempt in range(TRANSPORT_RETRIES):
            try:
                text, usage, meta = await call_once(client, model, prompt)
                break
            except Exception as e:  # noqa: BLE001 — network/HTTP errors
                err = f"{type(e).__name__}: {e}"
                if attempt < TRANSPORT_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt + random.random())
        parsed = parse_verdict(text) if not err else None
        if not err and parsed is None:
            # protocol: one re-ask on invalid JSON
            try:
                text2, usage2, meta2 = await call_once(client, model, prompt)
                parsed = parse_verdict(text2)
                if parsed is not None:
                    text, usage, meta = text2, usage2, meta2
            except Exception as e:  # noqa: BLE001
                err = f"reask {type(e).__name__}: {e}"
        record["latency_s"] = round(time.monotonic() - t0, 2)
        record["ts_end"] = utc_now()
        record["usage"] = usage
        record.update(meta)
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


CORPORA = {
    "en": (DATA / "pilot_sample.csv", "judge_runs", ("Teacher:", "Student:")),
    "ru": (RU / "ru_sample.csv", "judge_runs_ru", ("Учитель:", "Ученик:")),
    "native": (RU / "native_sample.csv", "judge_runs_native", ("Учитель:", "Ученик:")),
    "flawed": (RU / "native_flawed_sample.csv", "judge_runs_native_flawed", ("Учитель:", "Ученик:")),
}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--dialogs", type=int, default=0, help="limit to first N dialogs (0 = all)")
    ap.add_argument("--criteria", nargs="*", default=None, help="subset of criteria (default: all)")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--concurrency", type=int, default=int(os.environ.get("JUDGE_CONCURRENCY", "16")))
    ap.add_argument("--per-model", type=int, default=6, help="max in-flight calls per model")
    ap.add_argument("--lang", choices=list(CORPORA), default="en",
                    help="en: K1 (MathDial); ru: K2 (translation); native: K3; flawed: K4. "
                         "Judge instructions are always in English")
    ap.add_argument("--langs", nargs="*", choices=list(CORPORA), default=None,
                    help="several corpora in ONE run (overrides --lang); use with --interleave")
    ap.add_argument("--interleave", action="store_true",
                    help="shuffle all calls (models x corpora x units x repeats) with --seed")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--out-suffix", default="", help="write to <logs>/<runs_dir><suffix>/")
    args = ap.parse_args()

    load_env()
    from llm_client import make_client

    langs = args.langs or [args.lang]
    criteria = args.criteria or list(CRITERIA)
    specs = []  # (lang, model, pilot_id, criterion, repeat, prompt, out_path)
    for lang in langs:
        sample_file, runs_name, labels = CORPORA[lang]
        runs_dir = LOGS / f"{runs_name}{args.out_suffix}"
        runs_dir.mkdir(parents=True, exist_ok=True)
        sample = list(csv.DictReader(open(sample_file)))
        sample.sort(key=lambda r: r["pilot_id"])
        if args.dialogs:
            sample = sample[: args.dialogs]
        for model in args.models:
            out_path = runs_dir / f"raw_{model.replace('/', '_')}.jsonl"
            skip = done_keys(model, runs_dir)
            for r in sample:
                dialogue = (render_dialog(r["conversation"], strip_tags=True)
                            .replace("**Учитель:**", labels[0])
                            .replace("**Ученик:**", labels[1]))
                for criterion in criteria:
                    prompt = build_prompt(r["question"], r["ground_truth"],
                                          r["student_incorrect_solution"], dialogue, criterion)
                    for rep in range(1, args.repeats + 1):
                        if (r["pilot_id"], criterion, rep) in skip:
                            continue
                        specs.append((lang, model, r["pilot_id"], criterion, rep, prompt, out_path))
    if args.interleave:
        random.Random(args.seed).shuffle(specs)

    sem_global = asyncio.Semaphore(args.concurrency)
    sem_models = {m: asyncio.Semaphore(args.per_model) for m in args.models}
    clients = {m: make_client(m) for m in args.models}
    stats = {"done": 0, "errors": 0, "total": len(specs)}
    tasks = []
    for order, (lang, model, pid, crit, rep, prompt, out_path) in enumerate(specs):
        extra = {"lang": lang, "order": order,
                 "schedule": f"interleave(seed={args.seed})" if args.interleave else "sequential"}
        tasks.append(run_unit(clients[model], sem_global, sem_models[model], model,
                              (pid, crit, rep, prompt), out_path, stats, extra))
    print(f"scheduled: {len(tasks)} calls, models={args.models}, langs={langs}, "
          f"interleave={args.interleave} seed={args.seed}, concurrency={args.concurrency}, "
          f"per_model={args.per_model}", flush=True)
    await asyncio.gather(*tasks)
    print(f"finished: {stats['done']} calls, errors: {stats['errors']}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
