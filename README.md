# RuTutorDial-70

**The first open Russian-language corpus of tutoring dialogues with expert rubric annotations — plus the full pipeline for benchmarking LLM judges on pedagogical criteria.**

Companion artifact for the paper *«LLM-судьи для оценки обучающих диалогов на русском языке: точность, воспроизводимость и верификация инъецированных нарушений»* (LLM Judges for Russian Tutoring Dialogues: Accuracy, Reproducibility, and Verification of Injected Violations), submitted 2026.

## What is inside

### `data/` — corpora and gold labels

| File | Description |
|---|---|
| `native_sample.csv` | **K3**: 35 clean synthetic RU tutoring dialogues (two-agent role-play, MathDial protocol, problems from human-translated mGSM-ru) |
| `native_flawed_sample.csv` | **K4**: 35 RU dialogues with controlled violation injection (reveal / ignore / math / overload), target criterion known by construction |
| `gold_ru70.csv` | Reconciled human gold for K3+K4: 5 criteria × 70 dialogues, per-verdict provenance (agreed / reconciled), flaw metadata |
| `ru_sample.csv` | **K2**: Russian translation of the 50 MathDial test dialogues (parallel corpus to K1) |
| `pilot_sample.csv`, `gold_pilot.csv` | **K1**: 50 English MathDial dialogues (stratified) + reconciled human gold |
| `annotation_*_completed.csv` | Raw independent sheets of both annotators (A/B), before reconciliation |
| `disagreements_*.csv` | Annotator disagreements with blind-reconciliation resolutions and per-case rules |
| `anon_mapping.csv` | Mapping of anonymous annotation IDs to corpus items (annotators were blind to corpus composition) |
| `rubric_v1.1_ru.md` | The 5-criterion pedagogical rubric, version 1.1 (Russian) |

### `judges/` — judge verdicts and summaries

Aggregated majority verdicts and result tables for the 8-judge panel (gpt-5.5, claude-opus-4.6, gemini-3-flash, qwen3-5-397b, deepseek-v3.2, gpt-oss-120b, gpt-oss-20b, gemma-3-27b; 3 repeats each, temperature 0). Raw per-call JSONL logs (~30 MB, 20 630 calls) are available on request / in a release archive.

### `code/` — full pipeline

Corpus construction (`make_pilot.py`, `translate_pilot.py`, `gen_native_ru.py`), judge runner with resume-safe logging (`run_judges.py`, `criteria_prompts.py`), agreement and gold-building (`interrater.py`, `build_gold*.py`), analysis (`aggregate_judges.py`, `summary_3corpora.py`, `compare_langs.py`).

Any OpenAI-compatible endpoint works:

```bash
export LLM_BASE_URL=https://openrouter.ai/api/v1   # or any compatible endpoint
export LLM_API_KEY=...
python code/run_judges.py --lang native --models gpt-5.5 --repeats 3
```

## Key findings (see paper)

1. Switching dialogue language EN→RU does **not** reduce judge accuracy (|Δ| ≤ 0.02 on a parallel corpus) but reduces verdict **reproducibility** (up to 21% flipped verdicts for smaller judges).
2. On human-verified gold, frontier judges reach **recall 0.93 at precision 0.78–0.80** across five pedagogical criteria.
3. **Prompt-based violation injection can silently fail**: the "ignore the student's misconception" flaw manifested in only 1/9 dialogues (human-verified) — construction-truth without manifestation checks yields false conclusions about judges.

## Licenses

- **Data** (`data/`): [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) — inherited from [MathDial](https://github.com/eth-nlped/mathdial) (Macina et al., 2023); problems in K3/K4 from [mGSM](https://arxiv.org/abs/2210.03057) (Shi et al., 2022).
- **Code** (`code/`): MIT.

## Аннотация (RU)

RuTutorDial-70 — первый открытый русскоязычный корпус тьюторских диалогов с эталонной разметкой по педагогической рубрике: 35 бездефектных и 35 диалогов с контролируемой инъекцией нарушений, эталон двух независимых аннотаторов со слепой сверкой, а также параллельный англо-русский корпус на базе MathDial и полный конвейер оценки LLM-судей (8 моделей, 6 800 юнит-вердиктов). Подробности — в статье и файлах `judges/summary_*.md`.

## Citation

```
Lazutkin D., Levinskaya M. LLM Judges for Russian Tutoring Dialogues:
Accuracy, Reproducibility, and Verification of Injected Violations. 2026.
[venue TBD]
```
