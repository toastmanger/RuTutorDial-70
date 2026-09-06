#!/usr/bin/env python3
"""Сводная матрица трёх корпусов: EN / RU-MT (перевод) / RU-native (синтез).

EN vs RU-MT — параллельные диалоги (эталон переносится): флипы, согласие с голдом.
Native — голда пока нет: fail-профили судей + попарное согласие судей по корпусам.
Выход: pilot/summary_3corpora.md
"""
import csv, glob, json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CRITS = ["C1_math", "C2_reveal", "C3_misconception", "C4_scaffolding", "C5_final_verdict"]
GM = {"1": "pass", "0": "fail", "NA": "na"}

def majority(dirname):
    units = defaultdict(list)
    for p in glob.glob(str(ROOT / "pilot" / dirname / "raw_*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            if not r.get("error"):
                units[(r["model"], r["pilot_id"], r["criterion"])].append(r["verdict"])
    return {k: Counter(v).most_common(1)[0][0] for k, v in units.items()}

en, ru, nat = majority("judge_runs"), majority("judge_runs_ru"), majority("judge_runs_native")
gold = {r["pilot_id"]: r for r in csv.DictReader(open(ROOT / "pilot" / "gold_pilot.csv"))}
models = sorted({k[0] for k in en})
d50 = [f"D{i:02d}" for i in range(1, 51)]
n35 = sorted({k[1] for k in nat})

L = ["# Сводная матрица: EN / RU-перевод / RU-натив (8 судей)\n",
     "_Эталон: v1.1 до точечной проверки; native — без человеческого эталона (ждёт разметки)._\n",
     "## 1. Параллельные корпуса EN vs RU-MT (50 диалогов, эталон общий)\n",
     "| Судья | Флипы EN→RU | Голд EN | Голд RU | Δ |", "|---|---|---|---|---|"]
crit_flips, crit_n, crit_agr = Counter(), Counter(), defaultdict(lambda: [0, 0])
for m in models:
    fl = aen = aru = n = 0
    for d in d50:
        for c in CRITS:
            ve, vr = en.get((m, d, c)), ru.get((m, d, c))
            if not ve or not vr: continue
            g = GM[gold[d][c]]
            n += 1; fl += ve != vr; aen += ve == g; aru += vr == g
            crit_flips[c] += ve != vr; crit_n[c] += 1
            crit_agr[c][0] += ve == g; crit_agr[c][1] += vr == g
    L.append(f"| {m} | {fl/n:.0%} | {aen/n:.2f} | {aru/n:.2f} | {(aru-aen)/n:+.2f} |")
L += ["\n### По критериям (все судьи)\n",
      "| Критерий | Флипы | Голд EN | Голд RU | Δ |", "|---|---|---|---|---|"]
for c in CRITS:
    a_en, a_ru = crit_agr[c][0]/crit_n[c], crit_agr[c][1]/crit_n[c]
    L.append(f"| {c} | {crit_flips[c]/crit_n[c]:.0%} | {a_en:.2f} | {a_ru:.2f} | {a_ru-a_en:+.2f} |")

L += ["\n## 2. Fail-доли судей по корпусам (профиль строгости)\n",
      "| Судья | " + " | ".join(f"{c.split('_')[0]} EN/RU/NAT" for c in CRITS) + " |",
      "|---|" + "---|" * len(CRITS)]
for m in models:
    cells = []
    for c in CRITS:
        fr = []
        for verd, ids in ((en, d50), (ru, d50), (nat, n35)):
            vs = [verd.get((m, d, c)) for d in ids]
            vs = [v for v in vs if v]
            fr.append(f"{sum(v=='fail' for v in vs)/len(vs):.2f}" if vs else "—")
        cells.append("/".join(fr))
    L.append(f"| {m} | " + " | ".join(cells) + " |")

L += ["\n## 3. Среднее попарное согласие судей (панельный сигнал)\n",
      "| Корпус | Согласие |", "|---|---|"]
for name, verd, ids in (("EN", en, d50), ("RU-MT", ru, d50), ("RU-native", nat, n35)):
    agrs = []
    for m1, m2 in combinations(models, 2):
        pair = [(verd.get((m1, d, c)), verd.get((m2, d, c))) for d in ids for c in CRITS]
        pair = [(a, b) for a, b in pair if a and b]
        if pair:
            agrs.append(sum(a == b for a, b in pair) / len(pair))
    L.append(f"| {name} | {sum(agrs)/len(agrs):.2f} |")

out = ROOT / "pilot" / "summary_3corpora.md"
out.write_text("\n".join(L))
print("\n".join(L))

# --- Секция 4: recall насаженных нарушений (дефектный корпус) ---
flawed_rows = list(csv.DictReader(open(ROOT / "pilot" / "ru" / "native_flawed_sample.csv")))
flw = majority("judge_runs_native_flawed")
flaw_types = ["reveal", "ignore", "math", "overload"]
L2 = ["\n## 4. Recall насаженных нарушений (истинность по построению)\n",
      "| Судья | " + " | ".join(flaw_types) + " | Итого |", "|---|" + "---|" * (len(flaw_types) + 1)]
for m in models:
    cells, th, tn = [], 0, 0
    for ft in flaw_types:
        hits = n = 0
        for r in [x for x in flawed_rows if x["flaw"] == ft]:
            v = flw.get((m, r["pilot_id"], r["expected_violation"]))
            if v:
                n += 1; hits += v == "fail"
        cells.append(f"{hits}/{n}")
        th += hits; tn += n
    L2.append(f"| {m} | " + " | ".join(cells) + f" | {th/tn:.0%} |")
with open(out, "a") as f:
    f.write("\n".join(L2))
print("\n".join(L2))
