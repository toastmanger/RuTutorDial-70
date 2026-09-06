#!/usr/bin/env python3
"""Build the reconciled pilot gold standard.

gold = sheet A verdicts patched with resolutions from disagreements_AB.csv.
Also prints spot-check candidates: agreed units potentially affected by the
new v1.1 rules (stricter per-step attempt counting in C2).
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT / "pilot"
CRITS = ["C1_math", "C2_reveal", "C3_misconception", "C4_scaffolding", "C5_final_verdict"]


def main() -> None:
    A = {r["pilot_id"]: r for r in csv.DictReader(open(PILOT / "annotation_sheet_A_completed.csv"))}
    B = {r["pilot_id"]: r for r in csv.DictReader(open(PILOT / "annotation_sheet_B_completed.csv"))}
    res = list(csv.DictReader(open(PILOT / "disagreements_AB.csv")))
    sample = {r["pilot_id"]: r for r in csv.DictReader(open(PILOT / "pilot_sample.csv"))}

    unresolved = [r for r in res if r["resolution"].strip() not in {"0", "1"}]
    assert not unresolved, f"unresolved cases: {[(r['pilot_id'], r['criterion']) for r in unresolved]}"

    patch = {(r["pilot_id"], r["criterion"]): r["resolution"].strip() for r in res}
    gold_rows = []
    for d in sorted(A):
        row = {"pilot_id": d, "stratum": sample[d]["self-correctness"]}
        for c in CRITS:
            v = patch.get((d, c), A[d][c].strip())
            row[c] = v
            row[c + "_source"] = "reconciled" if (d, c) in patch else "agreed"
        gold_rows.append(row)

    with open(PILOT / "gold_pilot.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(gold_rows[0].keys()))
        w.writeheader()
        w.writerows(gold_rows)

    # how often did reconciliation side with A vs B?
    a_wins = sum(1 for r in res if r["resolution"].strip() == r["verdict_A"].strip())
    print(f"gold_pilot.csv: {len(gold_rows)} dialogues; reconciled units: {len(res)} "
          f"(sided with A: {a_wins}/{len(res)})")

    # spot-check candidates for the stricter C2 rule:
    # dialogues where the teacher self-reported revealing the answer, yet gold C2=1
    c2_candidates = [r["pilot_id"] for r in gold_rows
                     if r["C2_reveal"] == "1" and r["stratum"].startswith("Yes, but")]
    print("C2 spot-check candidates (self-reported reveal, gold C2=1):",
          ", ".join(c2_candidates) or "none")

    # C3: agreed '1' under the now-explicit strict rule — random 5 for QC
    import random
    c3_agreed_pass = [r["pilot_id"] for r in gold_rows
                      if r["C3_misconception"] == "1" and r["C3_misconception_source"] == "agreed"]
    qc = sorted(random.Random(20260905).sample(c3_agreed_pass, min(5, len(c3_agreed_pass))))
    print(f"C3 QC sample (5 of {len(c3_agreed_pass)} agreed passes):", ", ".join(qc))


if __name__ == "__main__":
    main()
