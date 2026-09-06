#!/usr/bin/env python3
"""Interrater agreement A vs B on the pilot + disagreement list for reconciliation.

Outputs:
  pilot/interrater_report.md   — raw agreement + Cohen's kappa per criterion
  pilot/disagreements_AB.csv   — one row per (dialogue, criterion) where A != B,
                                 with both verdicts and both annotators' notes
"""
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT / "pilot"
CRITS = ["C1_math", "C2_reveal", "C3_misconception", "C4_scaffolding", "C5_final_verdict"]


def load(name: str) -> dict:
    rows = list(csv.DictReader(open(PILOT / name)))
    return {r["pilot_id"]: r for r in rows}


def kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    cats = {v for p in pairs for v in p}
    pa = Counter(a for a, _ in pairs)
    pb = Counter(b for _, b in pairs)
    pe = sum((pa[c] / n) * (pb[c] / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main() -> None:
    A, B = load("annotation_sheet_A_completed.csv"), load("annotation_sheet_B_completed.csv")
    assert set(A) == set(B) == {f"D{i:02d}" for i in range(1, 51)}

    lines = ["# Interrater A vs B — пилот 50 диалогов\n",
             "| Критерий | Raw agreement | Cohen's κ | Расхождений |",
             "|---|---|---|---|"]
    disagreements = []
    for c in CRITS:
        pairs = [(A[d][c].strip(), B[d][c].strip()) for d in sorted(A)]
        raw = sum(1 for a, b in pairs if a == b) / len(pairs)
        k = kappa(pairs)
        dis = [d for d in sorted(A) if A[d][c].strip() != B[d][c].strip()]
        lines.append(f"| {c} | {raw:.2f} | {k:.2f} | {len(dis)} |")
        for d in dis:
            disagreements.append({
                "pilot_id": d, "criterion": c,
                "verdict_A": A[d][c].strip(), "verdict_B": B[d][c].strip(),
                "notes_A": (A[d].get("notes") or "").strip(),
                "notes_B": (B[d].get("notes") or "").strip(),
                "resolution": "", "resolution_comment": "",
            })

    total_units = len(CRITS) * 50
    lines.append(f"\nВсего юнитов: {total_units}, расхождений: {len(disagreements)} "
                 f"({len(disagreements)/total_units:.1%})")
    lines.append("\nГейт протокола: raw agreement < 0.7 по критерию → критерий переопределять.")
    (PILOT / "interrater_report.md").write_text("\n".join(lines))

    with open(PILOT / "disagreements_AB.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(disagreements[0].keys()))
        w.writeheader()
        w.writerows(disagreements)

    print("\n".join(lines))


if __name__ == "__main__":
    main()
