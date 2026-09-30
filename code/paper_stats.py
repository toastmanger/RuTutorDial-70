#!/usr/bin/env python3
"""Statistics for v0.5: every number of the paper recomputed from raw judge logs,
with interval estimates and tests.

Unit u = (dialogue, criterion); all expectations E_u use equal weights w_u = 1/|U|.

Inference
- 95% CIs: stratified cluster bootstrap over dialogues (a dialogue carries all five
  criteria), percentile intervals, B = 10 000, seed 20260930. The bootstrap is paired:
  the same resampled dialogues are used for every judge and for both language versions,
  so differences (EN vs RU, judge vs judge) get CIs from the same resamples.
- McNemar exact test (binomial on discordant units) for EN vs RU correctness of the
  majority verdict and for judge vs judge on K3+K4.
- Permutation test for delta (H0: p_u = q_u): within each unit the 2k verdicts are
  re-split into two groups of k. Valid only when repeats are exchangeable in time,
  i.e. for the interleaved control run.
- Multiplicity: Holm (FWER) and Benjamini-Hochberg (FDR) over the family of 8 judges
  (28 pairs for judge vs judge).
- Cost: public per-token list prices (OpenRouter, fetched 2026-09-30); wall-clock from
  JSONL timestamps.

Output: <reports>/paper_stats.md and paper_stats.json (see layout.py)
"""
from __future__ import annotations

import csv
import glob
import gzip
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from layout import DATA, LOGS, REPORTS, RU  # noqa: E402

SEED = 20260930
B = 10_000
R_PERM = 10_000
CRITS = ["C1_math", "C2_reveal", "C3_misconception", "C4_scaffolding", "C5_final_verdict"]
GM = {"1": "pass", "0": "fail", "NA": "na"}
SHORT = {
    "gpt-5.5": "gpt-5.5",
    "anthropic/claude-opus-4.6": "claude-opus-4.6",
    "google/gemini-3-flash-preview": "gemini-3-flash",
    "qwen3-5-397b-a17b-fp8": "qwen3-5-397b",
    "gpt-oss-120b": "gpt-oss-120b",
    "google/gemma-3-27b-it": "gemma-3-27b",
    "openai/gpt-oss-20b": "gpt-oss-20b",
    "deepseek/deepseek-v3.2": "deepseek-v3.2",
}
# USD per 1M tokens (input, output); openrouter.ai/api/v1/models, fetched 2026-09-30
PRICES = {
    "gpt-5.5": (5.00, 30.00),
    "anthropic/claude-opus-4.6": (5.00, 25.00),
    "google/gemini-3-flash-preview": (0.50, 3.00),
    "qwen3-5-397b-a17b-fp8": (0.55, 3.50),
    "gpt-oss-120b": (0.037, 0.17),
    "google/gemma-3-27b-it": (0.08, 0.45),
    "openai/gpt-oss-20b": (0.018, 0.09),
    "deepseek/deepseek-v3.2": (0.28, 0.42),
    "moonshotai/kimi-k2.5": (0.45, 2.25),
}


# ---------------------------------------------------------------- loading
def log_files(dirname: str) -> list[str]:
    """raw_<model>.jsonl (working repository) or raw_<model>.jsonl.gz (artifact)."""
    return sorted(glob.glob(str(LOGS / dirname / "raw_*.jsonl")) + glob.glob(str(LOGS / dirname / "raw_*.jsonl.gz")))


def read_jsonl(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_runs(dirname: str) -> dict:
    """(model, pilot_id, criterion) -> successful records, in file (completion) order."""
    units = defaultdict(list)
    for p in log_files(dirname):
        for r in read_jsonl(p):
            if not r.get("error"):
                units[(r["model"], r["pilot_id"], r["criterion"])].append(r)
    return units


def load_all_records(dirname: str) -> list[dict]:
    out = []
    for p in log_files(dirname):
        out += read_jsonl(p)
    return out


gold_k1 = {r["pilot_id"]: r for r in csv.DictReader(open(DATA / "gold_pilot.csv"))}
gold_k34 = {r["pilot_id"]: r for r in csv.DictReader(open(RU / "gold_ru70.csv"))}
K1_IDS = sorted(gold_k1)
K3_IDS = sorted(p for p in gold_k34 if p.startswith("N"))
K4_IDS = sorted(p for p in gold_k34 if p.startswith("F"))


def majority(verdicts: list[str]) -> str:
    # tie-break = first verdict in completion order (as in the v0.3/v0.4 scripts)
    return Counter(verdicts).most_common(1)[0][0]


def disc_within(vs: list[str]) -> float:
    pairs = list(combinations(range(len(vs)), 2))
    return sum(vs[i] != vs[j] for i, j in pairs) / len(pairs)


def disc_cross(a: list[str], b: list[str]) -> float:
    return sum(x != y for x in a for y in b) / (len(a) * len(b))


# ---------------------------------------------------------------- inference helpers
def boot_index(strata: list[list[int]], rng) -> np.ndarray:
    """(B, n_dialogs) matrix of resampled dialogue indices, stratified."""
    cols = [rng.choice(np.asarray(s), size=(B, len(s)), replace=True) for s in strata]
    return np.concatenate(cols, axis=1)


def ratio_ci(num: np.ndarray, den: np.ndarray, idx: np.ndarray):
    """Point estimate and percentile CI of sum(num)/sum(den) under dialogue resampling."""
    point = num.sum() / den.sum() if den.sum() else float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    bs = bs[np.isfinite(bs)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def mcnemar(a: np.ndarray, b: np.ndarray):
    """a, b: boolean correctness on the same units. Returns (n10, n01, exact p)."""
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    p = stats.binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return n10, n01, float(p)


def holm(p: list[float]) -> list[float]:
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj.tolist()


def bh(p: list[float]) -> list[float]:
    m = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    run = 1.0
    for k, i in enumerate(order):
        rank = m - k
        run = min(run, p[i] * m / rank)
        adj[i] = run
    return adj.tolist()


def jeffreys(x: int, n: int):
    lo, hi = stats.beta.ppf([0.025, 0.975], x + 0.5, n - x + 0.5)
    return float(0.0 if x == 0 else lo), float(1.0 if x == n else hi)


def fmt_ci(t, d=2):
    return f"{t[0]:.{d}f} [{t[1]:.{d}f}; {t[2]:.{d}f}]"


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ---------------------------------------------------------------- per-unit tables
def unit_table(runs: dict, model: str, ids: list[str], gold: dict):
    """Per-unit arrays for one judge on one corpus (dialogue-major order)."""
    rows = []
    for d in ids:
        for c in CRITS:
            recs = runs.get((model, d, c), [])
            vs = [r["verdict"] for r in recs]
            if len(vs) < 2:
                raise SystemExit(f"{model} {d} {c}: only {len(vs)} repeats")
            g = GM[gold[d][c]]
            maj = majority(vs)
            rows.append({
                "dialog": d, "crit": c, "gold": g, "vs": vs, "maj": maj,
                "corr": maj == g, "a1": sum(v == g for v in vs) / len(vs),
                "cons": float(len(set(vs)) == 1), "din": disc_within(vs),
                "tie3": len(set(vs)) == 3,
            })
    return rows


def per_dialog(rows, key, ids):
    pos = {d: i for i, d in enumerate(ids)}
    num = np.zeros(len(ids))
    den = np.zeros(len(ids))
    for r in rows:
        v = key(r)
        if v is None:
            continue
        n, dd = v
        num[pos[r["dialog"]]] += n
        den[pos[r["dialog"]]] += dd
    return num, den


# ================================================================= main
def main():
    rng = np.random.default_rng(SEED)
    out = {"meta": {"seed": SEED, "B": B, "R_perm": R_PERM, "weights": "uniform 1/|U|"}}
    md = ["# Статистика v0.5 (все числа из сырых журналов)\n",
          f"_Бутстрэп: кластерный стратифицированный по диалогам, B={B}, сид {SEED}, перцентильные 95%-ДИ; "
          f"веса юнитов равные._\n"]

    orig = {"en": load_runs("judge_runs"), "ru": load_runs("judge_runs_ru"),
            "native": load_runs("judge_runs_native"), "flawed": load_runs("judge_runs_native_flawed")}
    k34 = dict(orig["native"])
    k34.update(orig["flawed"])
    models = list(SHORT)

    # ------------------------------------------------ Table 1: K3+K4
    ids34 = K3_IDS + K4_IDS
    strata34 = [list(range(len(K3_IDS))), list(range(len(K3_IDS), len(ids34)))]
    idx34 = boot_index(strata34, rng)
    t1 = {}
    corr34 = {}
    ties = Counter()
    for m in models:
        rows = unit_table(k34, m, ids34, gold_k34)
        ties[m] = sum(r["tie3"] for r in rows)
        corr34[m] = np.array([r["corr"] for r in rows])
        agree = ratio_ci(*per_dialog(rows, lambda r: (r["corr"], 1), ids34), idx34)
        a1 = ratio_ci(*per_dialog(rows, lambda r: (r["a1"], 1), ids34), idx34)
        gain = ratio_ci(*per_dialog(rows, lambda r: (r["corr"] - r["a1"], 1), ids34), idx34)
        prec = ratio_ci(*per_dialog(rows, lambda r: (r["gold"] == "fail", 1) if r["maj"] == "fail" else None, ids34), idx34)
        rec = ratio_ci(*per_dialog(rows, lambda r: (r["maj"] == "fail", 1) if r["gold"] == "fail" else None, ids34), idx34)
        cons = ratio_ci(*per_dialog(rows, lambda r: (r["cons"], 1), ids34), idx34)
        tp = {c: (sum(1 for r in rows if r["crit"] == c and r["gold"] == "fail" and r["maj"] == "fail"),
                  sum(1 for r in rows if r["crit"] == c and r["gold"] == "fail")) for c in CRITS}
        t1[m] = {"agree": agree, "a1": a1, "gain": gain, "precision": prec, "recall": rec, "C": cons,
                 "tp_by_crit": tp, "n_units": len(rows),
                 "tp": sum(v[0] for v in tp.values()), "pos": sum(v[1] for v in tp.values())}
    out["table1"] = t1
    out["table1_ties3"] = dict(ties)
    md += ["## Таблица 1. K3+K4 (350 юнитов), класс «нарушение» положительный\n",
           "| Судья | Согласие | $a_1$ | Выигрыш большинства | Точность | Полнота | $C$ |",
           "|---|---|---|---|---|---|---|"]
    for m in models:
        r = t1[m]
        md.append(f"| {SHORT[m]} | {fmt_ci(r['agree'])} | {fmt_ci(r['a1'])} | {fmt_ci(r['gain'], 3)} | "
                  f"{fmt_ci(r['precision'])} | {fmt_ci(r['recall'])} | {fmt_ci(r['C'])} |")
    md.append(f"\nТрёхсторонние ничьи (pass/fail/na в трёх повторах): {dict(ties)}\n")

    # counters: gold violations K3+K4 and detections
    viol = {c: {"K3": 0, "K4_target": 0, "K4_collateral": 0} for c in CRITS}
    for d in ids34:
        row = gold_k34[d]
        for c in CRITS:
            if row[c] == "0":
                if d.startswith("N"):
                    viol[c]["K3"] += 1
                elif row["expected_violation"] == c:
                    viol[c]["K4_target"] += 1
                else:
                    viol[c]["K4_collateral"] += 1
    out["violations_k34"] = viol
    md += ["## Счётчики: нарушения по эталону K3+K4 и их обнаружение (вердикт большинства)\n",
           "| Критерий | Всего | K3 | K4 целевые | K4 сопутствующие | " + " | ".join(SHORT[m] for m in models) + " |",
           "|---|---|---|---|---|" + "---|" * len(models)]
    for c in CRITS:
        v = viol[c]
        tot = sum(v.values())
        md.append(f"| {c} | {tot} | {v['K3']} | {v['K4_target']} | {v['K4_collateral']} | " +
                  " | ".join(f"{t1[m]['tp_by_crit'][c][0]}/{t1[m]['tp_by_crit'][c][1]}" for m in models) + " |")
    md.append("| Итого | " + str(sum(sum(v.values()) for v in viol.values())) + " | | | | " +
              " | ".join(f"{t1[m]['tp']}/{t1[m]['pos']}" for m in models) + " |\n")

    # manifestation (Table 3) with Jeffreys
    man = {}
    for fl in ["math", "reveal", "overload", "ignore"]:
        ds = [d for d in K4_IDS if gold_k34[d]["flaw"] == fl]
        x = sum(1 for d in ds if gold_k34[d][gold_k34[d]["expected_violation"]] == "0")
        man[fl] = {"x": x, "n": len(ds), "jeffreys": jeffreys(x, len(ds))}
    out["manifestation"] = man
    md += ["## Таблица 3. Манифестация (Джеффрис 95%)\n", "| Дефект | x/n | ДИ |", "|---|---|---|"]
    for fl, v in man.items():
        md.append(f"| {fl} | {v['x']}/{v['n']} | [{v['jeffreys'][0]:.2f}; {v['jeffreys'][1]:.2f}] |")

    # judge vs judge McNemar on K3+K4
    pairs = list(combinations(models, 2))
    pj = []
    for a, b in pairs:
        n10, n01, p = mcnemar(corr34[a], corr34[b])
        pj.append({"a": a, "b": b, "n10": n10, "n01": n01, "p": p})
    ph, pb = holm([x["p"] for x in pj]), bh([x["p"] for x in pj])
    for x, h, q in zip(pj, ph, pb):
        x["p_holm"], x["p_bh"] = h, q
    out["judge_vs_judge"] = pj
    md += ["\n## Судья↔судья на K3+K4: Мак-Немар (правильность вердикта большинства), 28 пар\n",
           "| Пара | a верно, b нет | b верно, a нет | p | p Холм | p BH |", "|---|---|---|---|---|---|"]
    for x in sorted(pj, key=lambda x: x["p"]):
        md.append(f"| {SHORT[x['a']]} vs {SHORT[x['b']]} | {x['n10']} | {x['n01']} | {fmt_p(x['p'])} | "
                  f"{fmt_p(x['p_holm'])} | {fmt_p(x['p_bh'])} |")

    # ------------------------------------------------ Table 2: K1-K2 (original run)
    strata1 = defaultdict(list)
    for i, d in enumerate(K1_IDS):
        strata1[gold_k1[d]["stratum"]].append(i)
    idx1 = boot_index(list(strata1.values()), rng)

    def parallel_block(en_runs, ru_runs, label, perm: bool):
        res = {}
        for m in models:
            e = unit_table(en_runs, m, K1_IDS, gold_k1)
            r = unit_table(ru_runs, m, K1_IDS, gold_k1)
            ce, cr = np.array([u["corr"] for u in e]), np.array([u["corr"] for u in r])
            dcross = np.array([disc_cross(u["vs"], w["vs"]) for u, w in zip(e, r)])
            din_e, din_r = np.array([u["din"] for u in e]), np.array([w["din"] for w in r])
            delta_u = dcross - 0.5 * (din_e + din_r)
            flips = np.array([u["maj"] != w["maj"] for u, w in zip(e, r)])
            dl = [u["dialog"] for u in e]

            def pdl(vals):
                num, den = np.zeros(len(K1_IDS)), np.zeros(len(K1_IDS))
                pos = {d: i for i, d in enumerate(K1_IDS)}
                for d, v in zip(dl, vals):
                    num[pos[d]] += v
                    den[pos[d]] += 1
                return num, den

            n10, n01, p_mc = mcnemar(ce, cr)
            row = {
                "agree_en": ratio_ci(*pdl(ce.astype(float)), idx1),
                "agree_ru": ratio_ci(*pdl(cr.astype(float)), idx1),
                "delta_agree": ratio_ci(*pdl(cr.astype(float) - ce.astype(float)), idx1),
                "mcnemar": {"en_only": n10, "ru_only": n01, "p": p_mc},
                "flips": ratio_ci(*pdl(flips.astype(float)), idx1),
                "din_en": ratio_ci(*pdl(din_e), idx1),
                "din_ru": ratio_ci(*pdl(din_r), idx1),
                "dcross": ratio_ci(*pdl(dcross), idx1),
                "delta": ratio_ci(*pdl(delta_u), idx1),
                "C_en": float(np.mean([u["cons"] for u in e])),
                "C_ru": float(np.mean([w["cons"] for w in r])),
            }
            if perm:
                # exact enumeration of the C(2k, k) splits per unit, Monte Carlo over units
                k = len(e[0]["vs"])
                splits = list(combinations(range(2 * k), k))
                table = np.empty((len(e), len(splits)))
                for ui, (u, w) in enumerate(zip(e, r)):
                    pool = u["vs"][:k] + w["vs"][:k]
                    for si, s in enumerate(splits):
                        g1 = [pool[i] for i in s]
                        g2 = [pool[i] for i in range(2 * k) if i not in s]
                        table[ui, si] = disc_cross(g1, g2) - 0.5 * (disc_within(g1) + disc_within(g2))
                prng = np.random.default_rng(SEED + 1)
                pick = prng.integers(0, len(splits), size=(R_PERM, len(e)))
                null = table[np.arange(len(e))[None, :], pick].mean(axis=1)
                obs = float(delta_u.mean())
                row["p_perm"] = float((1 + np.sum(null >= obs - 1e-12)) / (R_PERM + 1))
            res[m] = row
        pm = [res[m]["mcnemar"]["p"] for m in models]
        for m, h, q in zip(models, holm(pm), bh(pm)):
            res[m]["mcnemar"]["p_holm"], res[m]["mcnemar"]["p_bh"] = h, q
        if perm:
            pp = [res[m]["p_perm"] for m in models]
            for m, h, q in zip(models, holm(pp), bh(pp)):
                res[m]["p_perm_holm"], res[m]["p_perm_bh"] = h, q
        return res

    def md_parallel(res, title, perm):
        lines = [f"## {title}\n",
                 "| Судья | EN | RU | Δ RU−EN | Мак-Немар (EN/RU-only), p, p Холм | Флипы | $D_{in}$ EN | $D_{in}$ RU | $D_{cross}$ | $\\delta$ |"
                 + (" p перест. (Холм; BH) |" if perm else ""),
                 "|---|---|---|---|---|---|---|---|---|---|" + ("---|" if perm else "")]
        for m in models:
            x = res[m]
            mc = x["mcnemar"]
            line = (f"| {SHORT[m]} | {fmt_ci(x['agree_en'])} | {fmt_ci(x['agree_ru'])} | {fmt_ci(x['delta_agree'])} | "
                    f"{mc['en_only']}/{mc['ru_only']}, {fmt_p(mc['p'])}, {fmt_p(mc['p_holm'])} | {x['flips'][0]:.2f} | "
                    f"{fmt_ci(x['din_en'], 3)} | {fmt_ci(x['din_ru'], 3)} | {fmt_ci(x['dcross'], 3)} | {fmt_ci(x['delta'], 3)} |")
            if perm:
                line += f" {fmt_p(x['p_perm'])} ({fmt_p(x['p_perm_holm'])}; {fmt_p(x['p_perm_bh'])}) |"
            lines.append(line)
        return lines

    t2 = parallel_block(orig["en"], orig["ru"], "original", perm=False)
    out["table2_original"] = t2
    md += md_parallel(t2, "Таблица 2 (исходный прогон 05–06.09, повторы подряд)", False)

    # ------------------------------------------------ control run (interleaved), if present
    ctrl_en, ctrl_ru = load_runs("judge_runs_control0930"), load_runs("judge_runs_ru_control0930")
    if ctrl_en and ctrl_ru and not os.environ.get("NO_CONTROL"):
        t2c = parallel_block(ctrl_en, ctrl_ru, "control", perm=True)
        out["table2_control"] = t2c
        md += md_parallel(t2c, "Таблица 2К (контрольный прогон 30.09, все вызовы перемешаны)", True)
        # simultaneity artefact: D_in original vs interleaved on the same units (paired bootstrap)
        md += ["\n## Одновременные повторы vs перемешанные: $D_{in}$ исходного и контрольного прогонов\n",
               "| Судья | $D_{in}$ EN исх. | $D_{in}$ EN контр. | $D_{in}$ RU исх. | $D_{in}$ RU контр. | $\\delta$ исх. | $\\delta$ контр. |",
               "|---|---|---|---|---|---|---|"]
        for m in models:
            a, b = t2[m], t2c[m]
            md.append(f"| {SHORT[m]} | {a['din_en'][0]:.3f} | {b['din_en'][0]:.3f} | {a['din_ru'][0]:.3f} | "
                      f"{b['din_ru'][0]:.3f} | {a['delta'][0]:+.3f} | {b['delta'][0]:+.3f} |")
        # temporal drift: EN(05.09) vs EN(30.09), RU(06.09) vs RU(30.09) -- cross-discordance vs new D_in
        drift = {}
        for m in models:
            for lang, old, new in (("en", orig["en"], ctrl_en), ("ru", orig["ru"], ctrl_ru)):
                o = unit_table(old, m, K1_IDS, gold_k1)
                n = unit_table(new, m, K1_IDS, gold_k1)
                dc = np.mean([disc_cross(u["vs"], w["vs"]) for u, w in zip(o, n)])
                dt_u = np.array([disc_cross(u["vs"], w["vs"]) - 0.5 * (u["din"] + w["din"]) for u, w in zip(o, n)])
                pos = {d: i for i, d in enumerate(K1_IDS)}
                num, den = np.zeros(len(K1_IDS)), np.zeros(len(K1_IDS))
                for u, v in zip(o, dt_u):
                    num[pos[u["dialog"]]] += v
                    den[pos[u["dialog"]]] += 1
                drift_ci = ratio_ci(num, den, idx1)
                drift[f"{m}|{lang}"] = {"dcross_time": float(dc), "delta_time": drift_ci,
                                        "din_new": float(np.mean([w["din"] for w in n])),
                                        "din_old": float(np.mean([u["din"] for u in o])),
                                        "agree_old": float(np.mean([u["corr"] for u in o])),
                                        "agree_new": float(np.mean([w["corr"] for w in n]))}
        out["drift"] = drift
        md += ["\n## Дрейф во времени: тот же материал, 05/06.09 vs 30.09\n",
               "| Судья | язык | согласие исх. → контр. | $D_{cross}$(исх., контр.) | $D_{in}$ исх. | $D_{in}$ контр. | $\\delta_{время}$ |",
               "|---|---|---|---|---|---|---|"]
        for key, v in drift.items():
            m, lang = key.split("|")
            md.append(f"| {SHORT[m]} | {lang} | {v['agree_old']:.2f} → {v['agree_new']:.2f} | {v['dcross_time']:.3f} | "
                      f"{v['din_old']:.3f} | {v['din_new']:.3f} | {fmt_ci(v['delta_time'], 3)} |")
        # schedule diagnostics of the control run
        recs = load_all_records("judge_runs_control0930") + load_all_records("judge_runs_ru_control0930")
        ok = [r for r in recs if not r.get("error")]
        gaps = defaultdict(list)
        for r in ok:
            gaps[(r["model"], r["lang"], r["pilot_id"], r["criterion"])].append(
                datetime.fromisoformat(r["ts_start"].replace("Z", "+00:00")).timestamp())
        mg = sorted(max(v) - min(v) for v in gaps.values() if len(v) > 1)
        served = Counter((r["model"], r.get("served_model")) for r in ok)
        out["control_schedule"] = {
            "calls": len(recs), "errors": len(recs) - len(ok),
            "repeat_gap_median_s": float(np.median(mg)), "repeat_gap_p10_s": float(np.percentile(mg, 10)),
            "share_gap_le_5s": float(np.mean(np.array(mg) <= 5)),
            "window": [min(r["ts_start"] for r in ok), max(r["ts_end"] for r in ok)],
            "served_models": {f"{k[0]} -> {k[1]}": v for k, v in served.items()},
            "finish_reasons": dict(Counter(r.get("finish_reason") for r in ok)),
        }
        md += ["\n## Расписание контрольного прогона\n", "```", json.dumps(out["control_schedule"], ensure_ascii=False, indent=1), "```"]

    # ------------------------------------------------ Appendix C numbers (K1-K2: control run if present)
    k12 = {"en": ctrl_en, "ru": ctrl_ru} if (ctrl_en and ctrl_ru and not os.environ.get("NO_CONTROL")) else {"en": orig["en"], "ru": orig["ru"]}
    k12_label = "контрольного прогона" if k12["en"] is ctrl_en else "основного прогона"
    md += [f"\n## C.1 По критериям (все судьи), EN vs RU {k12_label}\n",
           "| Критерий | Флипы | EN | RU | Δ |", "|---|---|---|---|---|"]
    c1 = {}
    for c in CRITS:
        fl = ae = ar = n = 0
        for m in models:
            e = {u["dialog"]: u for u in unit_table(k12["en"], m, K1_IDS, gold_k1) if u["crit"] == c}
            r = {u["dialog"]: u for u in unit_table(k12["ru"], m, K1_IDS, gold_k1) if u["crit"] == c}
            for d in K1_IDS:
                n += 1
                fl += e[d]["maj"] != r[d]["maj"]
                ae += e[d]["corr"]
                ar += r[d]["corr"]
        c1[c] = {"flips": fl / n, "en": ae / n, "ru": ar / n, "n": n}
        md.append(f"| {c} | {fl / n:.0%} | {ae / n:.3f} | {ar / n:.3f} | {(ar - ae) / n:+.3f} |")
    out["appC1"] = c1

    gold_share = {c: sum(gold_k1[d][c] == "0" for d in K1_IDS) / len(K1_IDS) for c in CRITS}
    gold_na = {c: sum(gold_k1[d][c] == "NA" for d in K1_IDS) / len(K1_IDS) for c in CRITS}
    out["gold_k1_fail_share"], out["gold_k1_na_share"] = gold_share, gold_na
    md += ["\n## C.2 Доли «нарушено» EN / RU / K3; эталон K1: " +
           ", ".join(f"{c.split('_')[0]} {v:.2f}" for c, v in gold_share.items()) +
           "; доля NA в эталоне K1: " + ", ".join(f"{c.split('_')[0]} {v:.2f}" for c, v in gold_na.items()) + "\n",
           "| Судья | " + " | ".join(c.split("_")[0] for c in CRITS) + " | max отношение к эталону K1 (EN) |",
           "|---|" + "---|" * (len(CRITS) + 1)]
    c2 = {}
    for m in models:
        cells, ratios = [], []
        for c in CRITS:
            fr = []
            for runs, ids in ((k12["en"], K1_IDS), (k12["ru"], K1_IDS), (orig["native"], K3_IDS)):
                vs = [majority([x["verdict"] for x in runs[(m, d, c)]]) for d in ids]
                fr.append(sum(v == "fail" for v in vs) / len(vs))
            cells.append("/".join(f"{x:.2f}" for x in fr))
            ratios.append(fr[0] / gold_share[c] if gold_share[c] else float("nan"))
            c2[f"{m}|{c}"] = fr
        md.append(f"| {SHORT[m]} | " + " | ".join(cells) + f" | {max(ratios):.1f} |")
    na_share = {}
    for m in models:
        for c in ["C5_final_verdict"]:
            allv = [x["verdict"] for d in K1_IDS for x in k12["en"][(m, d, c)]] + \
                   [x["verdict"] for d in K1_IDS for x in k12["ru"][(m, d, c)]]
            na_share[m] = sum(v == "na" for v in allv) / len(allv)
    out["appC2"], out["c5_na_share_k1k2_all_repeats"] = c2, na_share
    na_all = {}
    for m in models:
        vs = [x["verdict"] for runs in list(orig.values()) + [ctrl_en, ctrl_ru] for (mm, _, _), recs in runs.items() if mm == m for x in recs]
        na_all[m] = [sum(v == "na" for v in vs), len(vs)]
    out["na_count_all_logs"] = na_all
    md.append("\nВердикт NA во всех журналах (NA / всего): " + ", ".join(f"{SHORT[m]} {a}/{b}" for m, (a, b) in na_all.items()))
    md.append("\nДоля NA по C5 во всех повторах K1+K2: " + ", ".join(f"{SHORT[m]} {v:.3f}" for m, v in na_share.items()))

    # judge pairwise agreement (panel signal)
    def panel(runs, ids):
        vals = []
        for a, b in combinations(models, 2):
            s = [majority([x["verdict"] for x in runs[(a, d, c)]]) == majority([x["verdict"] for x in runs[(b, d, c)]])
                 for d in ids for c in CRITS]
            vals.append(np.mean(s))
        return float(np.mean(vals))
    out["panel_agreement"] = {"K1": panel(k12["en"], K1_IDS), "K2": panel(k12["ru"], K1_IDS),
                              "K3": panel(orig["native"], K3_IDS), "K3+K4": panel(k34, ids34)}
    t2k = out.get("table2_control", t2)
    k1agree = {m: t2k[m]["agree_en"][0] for m in models}
    out["k1_C_en"] = {m: t2k[m]["C_en"] for m in models}
    out["k1_C_ru"] = {m: t2k[m]["C_ru"] for m in models}
    out["k1_agree_range"] = [min(k1agree.values()), max(k1agree.values())]
    md += ["\n## Панельное согласие судей между собой\n", json.dumps(out["panel_agreement"]),
           f"\nСогласие с эталоном на K1: {min(k1agree.values()):.3f}–{max(k1agree.values()):.3f}"]

    # ------------------------------------------------ cost and wall-clock
    # wall-clock is reported as a throughput estimate from logged latencies: minutes per 1000 calls
    # with 6 requests in flight = 1000 * mean latency / 6 / 60 (the original runs mixed models under
    # different concurrency, so their raw time spans are not comparable across judges)
    def usage(r, key):
        return (r.get("usage") or {}).get(key, 0) or 0

    ORIG_DIRS = ["judge_runs", "judge_runs_ru", "judge_runs_native", "judge_runs_native_flawed"]
    CTRL_DIRS = ["judge_runs_control0930", "judge_runs_ru_control0930"]
    recs_by_dir = {d: load_all_records(d) for d in ORIG_DIRS + CTRL_DIRS}
    cost = {}
    for m in models:
        orig_recs = [r for d in ORIG_DIRS for r in recs_by_dir[d] if r["model"] == m]
        ctrl_recs = [r for d in CTRL_DIRS for r in recs_by_dir[d] if r["model"] == m]
        ok = [r for r in orig_recs + ctrl_recs if not r.get("error")]
        k34 = [r for d in ("judge_runs_native", "judge_runs_native_flawed") for r in recs_by_dir[d] if r["model"] == m]
        pi, po = PRICES[m]

        def usd(rs):
            return sum(usage(r, "prompt") * pi + usage(r, "completion") * po for r in rs) / 1e6

        pin = np.mean([usage(r, "prompt") for r in orig_recs])
        pout = np.mean([usage(r, "completion") for r in orig_recs])
        mean_lat = float(np.mean([r["latency_s"] for r in ok]))
        cost[m] = {
            "tokens_in_avg": float(pin), "tokens_out_avg": float(pout),
            "usd_per_1k_units_k1": float((pin * pi + pout * po) / 1e6 * 1000),
            "usd_ru70_1050_calls": float(usd(k34) / len(k34) * 1050),
            "usd_parallel_pair_1500_calls": float(usd(ctrl_recs) / len(ctrl_recs) * 1500) if ctrl_recs else None,
            "latency_median_s": float(np.median([r["latency_s"] for r in ok])),
            "latency_mean_s": mean_lat,
            "wall_min_per_1k_calls_at_6": mean_lat * 1000 / 6 / 60,
            "wall_min_ru70_at_6": mean_lat * 1050 / 6 / 60,
            "calls_original": len(orig_recs), "calls_control": len(ctrl_recs),
            "usd_original": float(usd(orig_recs)), "usd_control": float(usd(ctrl_recs)),
        }
    out["cost"] = cost
    ctrl_all = [r for d in CTRL_DIRS for r in recs_by_dir[d]]
    if ctrl_all:
        ts0 = min(r["ts_start"] for r in ctrl_all)
        ts1 = max(r.get("ts_end", r["ts_start"]) for r in ctrl_all)
        dt = (datetime.fromisoformat(ts1.replace("Z", "+00:00")) - datetime.fromisoformat(ts0.replace("Z", "+00:00"))).total_seconds()
        out["control_wall_clock_min"] = dt / 60
    md += ["\n## Стоимость и время (тарифы OpenRouter на 30.09.2026; время — оценка по средней задержке при 6 параллельных запросах)\n",
           "| Судья | ток. вход/выход на вызов | USD на 1000 юнитов (k=1) | USD: RuTutorDial-70 (1050 вызовов) | USD: K1–K2 (1500) | задержка медиана/среднее, с | мин на 1000 вызовов | мин на RuTutorDial-70 | USD исходные / контроль |",
           "|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        c = cost[m]
        pp = f"{c['usd_parallel_pair_1500_calls']:.2f}" if c["usd_parallel_pair_1500_calls"] is not None else "—"
        md.append(f"| {SHORT[m]} | {c['tokens_in_avg']:.0f}/{c['tokens_out_avg']:.0f} | {c['usd_per_1k_units_k1']:.2f} | "
                  f"{c['usd_ru70_1050_calls']:.2f} | {pp} | {c['latency_median_s']:.1f}/{c['latency_mean_s']:.1f} | "
                  f"{c['wall_min_per_1k_calls_at_6']:.1f} | {c['wall_min_ru70_at_6']:.1f} | {c['usd_original']:.2f} / {c['usd_control']:.2f} |")
    tot_o = sum(c["usd_original"] for c in cost.values()); tot_c = sum(c["usd_control"] for c in cost.values())
    n_o = sum(c["calls_original"] for c in cost.values()); n_c = sum(c["calls_control"] for c in cost.values())
    md.append(f"\nИтого: исходные прогоны {n_o} вызовов, ${tot_o:.2f}; контрольный {n_c} вызовов, ${tot_c:.2f}; "
              f"всего {n_o + n_c} вызовов, ${tot_o + tot_c:.2f}. "
              + (f"Контрольный прогон занял {out['control_wall_clock_min']:.0f} мин (8 судей параллельно, до 6 запросов на модель)."
                 if ctrl_all else ""))

    (REPORTS / "paper_stats.md").write_text("\n".join(md) + "\n")
    (REPORTS / "paper_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print("\n".join(md))


if __name__ == "__main__":
    main()
