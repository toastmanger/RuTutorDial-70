#!/usr/bin/env python3
"""Aggregate judge runs: majority vote, consistency, agreement vs pilot gold.

Outputs:
  pilot/judge_runs/verdicts.csv       — unit-level majority verdicts per judge
  pilot/judge_runs/summary.md         — Т3-заготовка: согласие с эталоном + консистентность
  pilot/judge_runs/disagreements_judge_gold.csv — юниты судья≠эталон (вход арбитража)

Note: эталон = gold_pilot.csv; до завершения точечной проверки задетых юнитов
(rubric-v1.1.md) все числа согласия — предварительные.
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "pilot" / "judge_runs"
CRITS = ["C1_math", "C2_reveal", "C3_misconception", "C4_scaffolding", "C5_final_verdict"]
GOLD_MAP = {"1": "pass", "0": "fail", "NA": "na"}


def load_records() -> dict:
    """(model, pilot_id, criterion, repeat) -> последняя не-ошибочная запись."""
    latest = {}
    for path in sorted(RUNS.glob("raw_*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error"):
                continue
            latest[(r["model"], r["pilot_id"], r["criterion"], r["repeat"])] = r
    return latest


def main() -> None:
    gold = {r["pilot_id"]: r for r in csv.DictReader(open(ROOT / "pilot" / "gold_pilot.csv"))}
    latest = load_records()

    units = defaultdict(list)  # (model, pilot_id, criterion) -> [verdicts]
    for (model, pid, crit, _rep), r in latest.items():
        units[(model, pid, crit)].append(r["verdict"])

    verdict_rows, models = [], sorted({k[0] for k in units})
    for (model, pid, crit), vs in sorted(units.items()):
        cnt = Counter(vs)
        top, n_top = cnt.most_common(1)[0]
        tie = list(cnt.values()).count(n_top) > 1
        verdict_rows.append({
            "model": model, "pilot_id": pid, "criterion": crit,
            "verdict": "tie" if tie else top,
            "n_repeats": len(vs), "n_agree": n_top,
            "consistent": int(len(cnt) == 1),
        })
    with open(RUNS / "verdicts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(verdict_rows[0].keys()))
        w.writeheader()
        w.writerows(verdict_rows)

    # summary + disagreements
    dis_rows = []
    lines = ["# Судьи vs эталон (пилот) — ПРЕДВАРИТЕЛЬНО (эталон до точечной проверки)\n",
             "| Судья | Юнитов | Консистентность 3/3 | Согласие с эталоном | По критериям (согласие) |",
             "|---|---|---|---|---|"]
    for model in models:
        rows = [r for r in verdict_rows if r["model"] == model]
        cons = sum(r["consistent"] for r in rows) / len(rows)
        agree_total, per_crit = [], []
        for crit in CRITS:
            crit_rows = [r for r in rows if r["criterion"] == crit]
            agree = 0
            for r in crit_rows:
                g = GOLD_MAP[gold[r["pilot_id"]][crit]]
                ok = (r["verdict"] == g)
                agree += ok
                agree_total.append(ok)
                if not ok:
                    dis_rows.append({
                        "model": model, "pilot_id": r["pilot_id"], "criterion": crit,
                        "judge": r["verdict"], "gold": g,
                        "consistent": r["consistent"],
                        "gold_source": gold[r["pilot_id"]][crit + "_source"],
                    })
            per_crit.append(f"{crit.split('_')[0]}:{agree/len(crit_rows):.2f}" if crit_rows else f"{crit}:—")
        lines.append(f"| {model} | {len(rows)} | {cons:.2f} | "
                     f"{sum(agree_total)/len(agree_total):.2f} | {' '.join(per_crit)} |")

    n_dis = len(dis_rows)
    n_units = len(verdict_rows)
    lines.append(f"\nРасхождений судья↔эталон: {n_dis} из {n_units} юнитов-вердиктов "
                 f"({n_dis/n_units:.1%}) — это вход арбитража.")
    by_crit = Counter(r["criterion"] for r in dis_rows)
    lines.append("Расхождения по критериям: " +
                 ", ".join(f"{c}: {by_crit.get(c, 0)}" for c in CRITS))
    (RUNS / "summary.md").write_text("\n".join(lines))

    with open(RUNS / "disagreements_judge_gold.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dis_rows[0].keys()))
        w.writeheader()
        w.writerows(dis_rows)

    print("\n".join(lines))


if __name__ == "__main__":
    main()
