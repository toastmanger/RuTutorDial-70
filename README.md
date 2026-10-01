# RuTutorDial-70

Russian-language tutoring dialogues with gold rubric labels, plus the complete code and raw logs for a verifiable reliability evaluation of LLM judges.

Companion artifact for the paper «Верифицируемая оценка надёжности LLM-судей на русскоязычных образовательных диалогах» (*Verifiable Reliability Evaluation of LLM Judges on Russian-Language Educational Dialogues*), D. V. Lazutkin, M. A. Levinskaya, 2026 (manuscript in preparation).

## Reproduce every table of the paper

```bash
python3.12 -m venv .venv && . .venv/bin/activate
make install      # pinned dependencies from requirements.txt
make reproduce    # -> judges/paper_stats.md and judges/paper_stats.json
```

`make reproduce` recomputes all tables of the paper from the raw logs in `logs/` with fixed seeds: point estimates, 95% cluster-bootstrap intervals, McNemar tests, permutation tests for the verdict-distribution shift δ, Holm and Benjamini–Hochberg adjustments, cost and throughput. It needs no API access.

## Contents

### `data/` — corpora and gold labels

| File | Description |
|---|---|
| `native_sample.csv` | **K3**: 35 clean synthetic RU tutoring dialogues (two-agent role-play following the MathDial protocol; problems from the human-translated Russian part of mGSM) |
| `native_flawed_sample.csv` | **K4**: 35 RU dialogues with controlled defect injection (reveal / ignore / math / overload); target criterion known by construction, manifestation verified by the gold labels |
| `gold_ru70.csv` | Reconciled human gold for K3+K4: 5 criteria × 70 dialogues, per-verdict provenance (agreed / reconciled), defect metadata |
| `pilot_sample.csv`, `gold_pilot.csv` | **K1**: 50 English MathDial test dialogues (stratified by outcome) + reconciled human gold |
| `ru_sample.csv` | **K2**: Russian translation of K1 (parallel corpus) |
| `annotation_*_completed.csv` | Independent sheets of both annotators (A/B), before reconciliation |
| `disagreements_*.csv` | Annotator disagreements with blind-reconciliation resolutions and the rule behind each |
| `anon_mapping.csv` | Anonymous annotation IDs → corpus items (annotators were blind to corpus composition) |
| `rubric_v1.1_ru.md` | The five-criterion pedagogical rubric, version 1.1 |

### `logs/` — raw judge logs

33 025 gzipped JSONL records, one per judge API call: 20 630 from the main runs and 12 395 from the interleaved control run (12 000 successful calls plus 395 calls rejected by the API under load and re-sent at the end of the run). The 628 records of failed attempts are kept for transparency. Records carry the call timestamps, model id (and, in the control run, the version id returned by the provider), unit, repeat, request parameters, verdict, evidence quote, reasoning, latency and token usage. Error records are kept; their messages are normalised (gateway request ids removed).

| Directory | Corpus | Date | Schedule |
|---|---|---|---|
| `judge_runs` | K1 (EN) | 2026-09-05 | repeats of a unit sent back-to-back |
| `judge_runs_ru` | K2 (RU translation) | 2026-09-06 | back-to-back |
| `judge_runs_native` | K3 | 2026-09-06 | back-to-back |
| `judge_runs_native_flawed` | K4 | 2026-09-06 | back-to-back |
| `judge_runs_control0930`, `judge_runs_ru_control0930` | K1 and K2 | 2026-09-30 | all 12 000 calls shuffled (seed 20260930): repeats and language versions interleaved |

Main-run records have `ts` (start time, Moscow time UTC+3, 1 s resolution); control-run records add `ts_start`/`ts_end` (UTC, 1 ms), `served_model`, `finish_reason`, `params`, `order`.

### `judges/` — reports

`paper_stats.md` / `paper_stats.json` — output of `make reproduce`, the numbers of the paper; `verdicts_en.csv` — majority verdicts of the main K1 run (5 Sep 2026); `archive_2026-09-06/` — working summaries of 6 Sep 2026 with earlier numbers, kept for provenance and superseded by `paper_stats.md`.

### `code/`

- `paper_stats.py` — every table and test of the paper (`make reproduce`).
- `run_judges.py` — judge runner: resume-safe, per-call JSONL, optional interleaved schedule (`--interleave --seed`), several corpora in one run (`--langs`).
- `criteria_prompts.py` — judge prompt template (Appendix B.1 of the paper) and English versions of the criterion definitions (Appendix A); `make_pilot.py` — dialogue rendering; `layout.py` — file layout.
- Construction scripts — `translate_pilot.py`, `gen_native_ru.py`, `build_gold.py`, `build_gold_ru70.py`, `interrater.py` — document how the corpora and the gold labels were built. They expect the working-repository layout (`pilot/…`) and inputs not redistributed here (e.g. the MathDial source tree).
- `aggregate_judges.py`, `summary_3corpora.py`, `compare_langs.py` — early analysis scripts, superseded by `paper_stats.py`.

## Re-running the judges

```bash
export LLM_BASE_URL=https://openrouter.ai/api/v1   # any OpenAI-compatible endpoint
export LLM_API_KEY=...
make judges                                        # interleaved EN/RU design, seed 20260930
```

`make judges` uses the OpenRouter ids of the eight judges (`MODELS` in the Makefile); for another endpoint pass your own: `make judges MODELS="..."`. The ids recorded in `logs/` are those of the API used in the study. Requests use temperature 0 (gpt-5.5 does not accept the parameter), no top_p / max_tokens, no API seed.

## Environment

Python 3.12 (tested with 3.12.13); numpy 2.5.3, scipy 1.18.1, openai 3.8.0, httpx 0.28.1 (`requirements.txt`).

## Licenses

- **Data and logs** (`data/`, `logs/`): [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) (`data/LICENSE`) — inherited from [MathDial](https://github.com/eth-nlped/mathdial) (Macina et al., 2023); problems in K3/K4 from [mGSM](https://arxiv.org/abs/2210.03057) (Shi et al., 2022).
- **Code** (`code/`): MIT (`LICENSE`).

## Аннотация (RU)

RuTutorDial-70 — открытый корпус русскоязычных тьюторских диалогов с эталонной разметкой по педагогической рубрике: 35 бездефектных и 35 диалогов с контролируемой инъекцией нарушений, эталон двух независимых аннотаторов со слепой сверкой, параллельный англо-русский корпус на базе MathDial, сырые журналы всех вызовов восьми LLM-судей и код, который одной командой воспроизводит все таблицы статьи.

## Citation

See `CITATION.cff`. The DOI of the archived Zenodo release is added here once the release is deposited.
