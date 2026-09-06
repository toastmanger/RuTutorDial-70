#!/usr/bin/env python3
"""EN vs RU judge verdicts on the same dialogues: flip rate + gold agreement.

Majority verdict per (model, pilot_id, criterion) from raw JSONL of both runs;
prints per-judge flip rate EN->RU, gold agreement in each language, and
per-criterion flip concentration (hard vs soft criteria interaction).
"""
import csv
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CRITS = ["C1_math", "C2_reveal", "C3_misconception", "C4_scaffolding", "C5_final_verdict"]
GM = {"1": "pass", "0": "fail", "NA": "na"}


def majority(dirpath: Path) -> dict:
    units = defaultdict(list)
    for p in glob.glob(str(dirpath / "raw_*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            if r.get("error"):
                continue
            units[(r["model"], r["pilot_id"], r["criterion"])].append(r["verdict"])
    return {k: Counter(v).most_common(1)[0][0] for k, v in units.items()}


def main() -> None:
    en = majority(ROOT / "pilot" / "judge_runs")
    ru = majority(ROOT / "pilot" / "judge_runs_ru")
    gold = {r["pilot_id"]: r for r in csv.DictReader(open(ROOT / "pilot" / "gold_pilot.csv"))}
    dialogs = sorted({k[1] for k in ru})
    models = sorted({k[0] for k in ru})

    print(f"{'судья':32} {'флипы EN→RU':>12} {'gold EN':>8} {'gold RU':>8}")
    tot_flips, tot_n = Counter(), Counter()
    for m in models:
        flips = agr_en = agr_ru = n = 0
        for d in dialogs:
            for c in CRITS:
                v_en, v_ru = en.get((m, d, c)), ru.get((m, d, c))
                if not v_en or not v_ru:
                    continue
                g = GM[gold[d][c]]
                n += 1
                flips += v_en != v_ru
                agr_en += v_en == g
                agr_ru += v_ru == g
                tot_flips[c] += v_en != v_ru
                tot_n[c] += 1
        print(f"{m:32} {flips:>4}/{n:<4} ({flips/n:.0%}) {agr_en/n:>7.2f} {agr_ru/n:>7.2f}")
    print("\nФлипы по критериям (все судьи):")
    for c in CRITS:
        print(f"  {c:20} {tot_flips[c]:>3}/{tot_n[c]:<4} ({tot_flips[c]/tot_n[c]:.0%})")


if __name__ == "__main__":
    main()
