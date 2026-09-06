#!/usr/bin/env python3
"""Голд RU-70: лист A + решения сверки (буква A/B -> вердикт выбранного разметчика).
Затем вскрытие anon-маппинга: манифестация дефектов по человеческому голду,
ложные срабатывания тьютора на чистом корпусе.
Выход: pilot/ru/gold_ru70.csv + печать сводки.
"""
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RU = ROOT / "pilot" / "ru"
CRITS = ["C1_math", "C2_reveal", "C3_misconception", "C4_scaffolding", "C5_final_verdict"]
LETTER = {"A": "A", "А": "A", "B": "B", "В": "B"}  # латиница и кириллица

A = {r["anon_id"]: r for r in csv.DictReader(open(RU / "annotation_ru70_A_completed.csv"))}
B = {r["anon_id"]: r for r in csv.DictReader(open(RU / "annotation_ru70_B_completed.csv"))}
res = list(csv.DictReader(open(RU / "disagreements_ru70_AB.csv")))
mapping = {r["anon_id"]: r for r in csv.DictReader(open(RU / "anon_mapping_SECRET.csv"))}

patch = {}
for r in res:
    w = LETTER.get(r["resolution"].strip().upper())
    assert w, f"нечитаемая resolution: {r['anon_id']} {r['criterion']} {r['resolution']!r}"
    src = A if w == "A" else B
    patch[(r["anon_id"], r["criterion"])] = src[r["anon_id"]][r["criterion"]].strip()

rows = []
for aid in sorted(A):
    m = mapping[aid]
    row = {"anon_id": aid, "pilot_id": m["pilot_id"], "flaw": m["flaw"],
           "expected_violation": m["expected_violation"]}
    for c in CRITS:
        row[c] = patch.get((aid, c), A[aid][c].strip())
        row[c + "_source"] = "reconciled" if (aid, c) in patch else "agreed"
    rows.append(row)
with open(RU / "gold_ru70.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

wins = Counter()
for r in res:
    wins[LETTER[r["resolution"].strip().upper()]] += 1
print(f"голд собран: 70 диалогов; сверка — в пользу A: {wins['A']}, в пользу B: {wins['B']}\n")

print("== Манифестация дефектов (человеческий голд: нарушен ли целевой критерий) ==")
for fl in ["reveal", "ignore", "math", "overload"]:
    ds = [r for r in rows if r["flaw"] == fl]
    hit = sum(1 for r in ds if r[r["expected_violation"]] == "0")
    print(f"  {fl:10} {hit}/{len(ds)} манифестировано")

print("\n== Чистый корпус (35): нарушения по мнению людей (ложные срабатывания генератора) ==")
clean = [r for r in rows if not r["flaw"]]
for c in CRITS:
    n0 = sum(1 for r in clean if r[c] == "0")
    print(f"  {c:20} {n0}/35 нарушений")

print("\n== Дефектные (35): нарушения НЕ по целевому критерию (сопутствующие) ==")
flawed = [r for r in rows if r["flaw"]]
side = Counter()
for r in flawed:
    for c in CRITS:
        if c != r["expected_violation"] and r[c] == "0":
            side[c] += 1
print(" ", dict(side))
