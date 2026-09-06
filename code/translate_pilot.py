#!/usr/bin/env python3
"""Translate pilot dialogues EN->RU via an OpenAI-compatible API.

Usage: .venv/bin/python scripts/translate_pilot.py [--dialogs 5]
Output: pilot/ru/ru_sample.csv — same columns as pilot_sample.csv, content in Russian,
format markers (|EOM|, Teacher:, Student:) preserved for downstream rendering.
Machine translation is the production pipeline (paper method: MT + human vetting);
vetting notes live in pilot/ru/vetting.md.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from run_judges import load_env  # noqa: E402

# Переводчик ВНЕ панели судей (self-preference confound): kimi-k2.5.
# gpt-5.5-версия первых 5 диалогов сохранена как sensitivity-check
# (pilot/ru/ru_sample_gpt55_5dialogs.csv).
MODEL = "moonshotai/kimi-k2.5"

PROMPT = """Translate the following math tutoring dialogue data from English to Russian.

STRICT RULES:
1. Preserve ALL numbers, calculations and math expressions exactly.
2. In the "conversation" field: keep the turn separator |EOM| and the speaker
   prefixes "Teacher:" / "Student:" EXACTLY as they are (do not translate the
   prefixes); translate only the content after them. Keep parenthesized tags
   like (probing) untouched if present.
3. Natural Russian school register: teacher addresses the student with «ты».
4. Translate names into natural transliteration (e.g. Julia -> Джулия).
5. Return strict JSON with the same keys: question, ground_truth,
   student_incorrect_solution, conversation. No other text.

DATA:
{data}"""


async def translate_row(client, row: dict) -> dict:
    from llm_client import normalize_model_id
    payload = {k: row[k] for k in
               ("question", "ground_truth", "student_incorrect_solution", "conversation")}
    last_err = None
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=normalize_model_id(MODEL),
                messages=[{"role": "user",
                           "content": PROMPT.format(data=json.dumps(payload, ensure_ascii=False))}],
            )
            text = resp.choices[0].message.content or ""
            m = re.search(r"\{.*\}", text, re.DOTALL)
            obj = json.loads(m.group(0), strict=False)
            out = dict(row)
            for k in payload:
                out[k] = obj[k]
            return out
        except Exception as e:  # noqa: BLE001 — таймауты/сеть/парсинг
            last_err = e
            await asyncio.sleep(2 ** attempt)
    raise last_err


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogs", type=int, default=50)
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    load_env()
    from llm_client import make_client

    sample = sorted(csv.DictReader(open(ROOT / "pilot" / "pilot_sample.csv")),
                    key=lambda r: r["pilot_id"])[: args.dialogs]
    outdir = ROOT / "pilot" / "ru"
    outdir.mkdir(exist_ok=True)
    raw_path = outdir / "ru_translation_raw.jsonl"
    done = set()
    if raw_path.exists():
        done = {json.loads(l)["pilot_id"] for l in raw_path.read_text().splitlines() if l.strip()}

    client = make_client(MODEL)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    async def worker(row):
        if row["pilot_id"] in done:
            return
        async with sem:
            try:
                out = await translate_row(client, row)
            except Exception as e:  # noqa: BLE001
                print(f"{row['pilot_id']} FAILED: {type(e).__name__}", flush=True)
                return
            async with lock:
                with open(raw_path, "a") as f:
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                print(f"{row['pilot_id']} ok", flush=True)

    await asyncio.gather(*[worker(r) for r in sample])

    recs = {}
    for l in raw_path.read_text().splitlines():
        r = json.loads(l)
        recs[r["pilot_id"]] = r
    rows = [recs[k] for k in sorted(recs)]
    with open(outdir / "ru_sample.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"translated {len(rows)}/{len(sample)} dialogues -> {outdir / 'ru_sample.csv'}")


if __name__ == "__main__":
    asyncio.run(main())
