#!/usr/bin/env python3
"""Build the pilot sample (50 dialogues) from MathDial test split.

Enriched (non-representative) design, fixed seeds — documented in pilot/pilot-protocol.md:
  20 x self-correctness == "Yes"
  15 x "Yes, but I had to reveal the answer"
  15 x "No"
Outputs into pilot/: pilot_sample.csv, dialogs.md, annotation_sheet_A.csv, annotation_sheet_B.csv
"""
import csv
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "mathdial-src" / "data" / "test.csv"
OUT = ROOT / "pilot"
SEED_SAMPLE = 20260905
SEED_SHUFFLE_A = 1
SEED_SHUFFLE_B = 2
STRATA = {
    "Yes": 20,
    "Yes, but I had to reveal the answer": 15,
    "No": 15,
}
MOVE_TAG = re.compile(r"^\((generic|focus|probing|telling)\)\s*")


def render_dialog(conversation: str, strip_tags: bool) -> str:
    lines = []
    for turn in conversation.split("|EOM|"):
        turn = turn.strip()
        if not turn:
            continue
        if turn.startswith("Teacher:"):
            body = turn[len("Teacher:"):].strip()
            if strip_tags:
                body = MOVE_TAG.sub("", body)
            lines.append(f"**Учитель:** {body}")
        elif turn.startswith("Student:"):
            lines.append(f"**Ученик:** {turn[len('Student:'):].strip()}")
        else:
            lines.append(turn)
    return "\n\n".join(lines)


def main() -> None:
    rows = [r for r in csv.DictReader(open(SRC)) if r["self-correctness"] in STRATA]
    # deterministic base order before sampling
    rows.sort(key=lambda r: (r["qid"], r["scenario"], r["conversation"][:50]))

    rng = random.Random(SEED_SAMPLE)
    pilot = []
    for stratum, n in STRATA.items():
        pool = [r for r in rows if r["self-correctness"] == stratum]
        pilot.extend(rng.sample(pool, n))
    rng.shuffle(pilot)  # mix strata in presentation order
    for i, r in enumerate(pilot, 1):
        r["pilot_id"] = f"D{i:02d}"

    OUT.mkdir(exist_ok=True)

    # 1) full sample with source fields
    with open(OUT / "pilot_sample.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pilot_id"] + [k for k in pilot[0] if k != "pilot_id"])
        w.writeheader()
        w.writerows(pilot)

    # 2) readable dialogues for annotators (move tags STRIPPED to avoid biasing)
    md = ["# Пилот — 50 диалогов MathDial (теги ходов учителя скрыты)\n"]
    for r in pilot:
        md.append(f"\n---\n\n## {r['pilot_id']}\n")
        md.append(f"**Задача:** {r['question']}\n")
        md.append(f"**Верный ответ:** {r['ground_truth'].splitlines()[-1] if r['ground_truth'] else '?'}\n")
        md.append(f"**Неверное решение ученика:**\n\n{r['student_incorrect_solution']}\n")
        md.append(f"\n{render_dialog(r['conversation'], strip_tags=True)}\n")
    (OUT / "dialogs.md").write_text("\n".join(md))

    # 3) annotation sheets, independent shuffles per annotator
    header = ["order", "pilot_id", "C1_math", "C2_reveal", "C3_misconception",
              "C4_scaffolding", "C5_final_verdict", "notes", "time_sec"]
    for name, seed in (("A", SEED_SHUFFLE_A), ("B", SEED_SHUFFLE_B)):
        ids = [r["pilot_id"] for r in pilot]
        random.Random(seed).shuffle(ids)
        with open(OUT / f"annotation_sheet_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for j, pid in enumerate(ids, 1):
                w.writerow([j, pid, "", "", "", "", "", "", ""])

    counts = {s: sum(1 for r in pilot if r["self-correctness"] == s) for s in STRATA}
    print(f"pilot: {len(pilot)} dialogues -> {OUT}")
    print("strata:", counts)
    print("median turns:", sorted(len(r['conversation'].split('|EOM|')) for r in pilot)[len(pilot)//2])


if __name__ == "__main__":
    main()
