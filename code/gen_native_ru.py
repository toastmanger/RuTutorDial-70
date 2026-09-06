#!/usr/bin/env python3
"""Generate native-Russian tutoring dialogues via two-agent self-play on mGSM-ru.

Protocol mirrors MathDial (Macina et al., 2023): a dialogue starts from a
plausible INCORRECT student solution; the tutor guides without revealing.
Both roles + misconception generation use ONE model OUTSIDE the judge panel
(kimi-k2.5) — no judge ever grades its own family's text, and generator style
is held constant across the translated and native RU corpora.

Usage: .venv/bin/python scripts/gen_native_ru.py [--n 35] [--concurrency 6]
Output: pilot/ru/native_raw.jsonl (resume-safe) + pilot/ru/native_sample.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from run_judges import load_env  # noqa: E402

MODEL = "moonshotai/kimi-k2.5"
MAX_TURNS = 14  # реплик всего (тьютор+ученик), потолок как в MathDial (медиана 11)
END_MARK = "[КОНЕЦ]"

SOLUTION_PROMPT = """Задача по математике для школьника:
{question}

Правильный ответ: {answer}

Придумай ПРАВДОПОДОБНОЕ НЕВЕРНОЕ решение этой задачи, какое мог бы написать
школьник {grade}. Ошибка должна быть типичной (неверная операция, потерянное
условие, двойной учёт, арифметический сбой) и вести к НЕВЕРНОМУ финальному ответу.
Реши пошагово от лица ученика, по-русски.

Верни строго JSON:
{{"incorrect_solution": "<пошаговое неверное решение с финальным числом>",
  "misconception": "<одно предложение: в чём именно ошибка>"}}"""

TUTOR_SYSTEM = """Ты — репетитор по математике, помогаешь школьнику разобраться с задачей,
которую он решил неверно. Говори по-русски, обращайся на «ты».

Задача: {question}
Правильный ответ (только для тебя, ученику не показывай): {answer}
Решение ученика (неверное): {incorrect_solution}

Правила:
- Веди ученика наводящими вопросами, один посильный шаг за раз, опирайся на уже установленное.
- НЕ сообщай готовый ответ и готовые шаги, пока ученик не сделал минимум две неудачные попытки на этом шаге.
- Найди конкретную ошибку в ЕГО решении и разбери с ним, почему это ошибка.
- Реплики короткие, 1-3 предложения, один вопрос за реплику.
- Когда ученик сам пришёл к правильному ответу — подтверди и добавь в конце реплики метку {end}."""

# Дефектные тьюторы: контролируемая инъекция нарушений (истинность по построению).
# flaw -> (ожидаемо нарушаемый критерий, добавка к роли вместо строгих правил)
FLAWS = {
    "reveal": ("C2_reveal", """- Ты нетерпеливый репетитор: быстро выдаёшь готовые шаги с числами
  и сам сообщаешь ответ, не дожидаясь двух попыток ученика на шаге. При этом дружелюбен."""),
    "ignore": ("C3_misconception", """- НЕ разбирай ошибку в решении ученика и не упоминай её.
  Просто веди его по своему правильному решению с нуля, шаг за шагом,
  как будто его решения не существовало."""),
    "math": ("C1_math", """- ОБЯЗАТЕЛЬНОЕ УСЛОВИЕ РОЛИ: во 2-й или 3-й своей реплике соверши
  математическую ошибку и НЕ исправляй её до конца диалога. Выбери одно:
  (а) назови результат вычисления, сдвинутый на 1-3 от верного, и уверенно на нём настаивай;
  (б) подтверди неверный шаг ученика словами «верно, молодец»;
  (в) отвергни верный шаг ученика как ошибочный.
  Диалог может закончиться неверным ответом — это нормально, так и задумано.
  В остальном веди себя как обычный доброжелательный репетитор."""),
    "overload": ("C4_scaffolding", """- Перегружай ученика: задавай по 2-3 вопроса в одной реплике,
  перескакивай через шаги, подставляй промежуточные результаты, которых ученик ещё не получал."""),
}

FLAWED_TUTOR_SYSTEM = """Ты — репетитор по математике, помогаешь школьнику с задачей,
которую он решил неверно. Говори по-русски, обращайся на «ты».

Задача: {question}
Правильный ответ (только для тебя): {answer}
Решение ученика (неверное): {incorrect_solution}

Твоя манера (следуй ей, это часть роли):
{flaw_rule}
- Реплики короткие, 1-3 предложения.
- Это ДИАЛОГ: после каждой твоей реплики жди ответа ученика. Не решай всё за одну реплику,
  даже если выдаёшь готовые шаги — выдавай их по одному и спрашивай реакцию.
- Метку {end} можно ставить не раньше твоей 4-й реплики, когда ученик согласился с итогом."""

STUDENT_SYSTEM = {
    "weak": """Ты — школьник, слабо понимающий математику. Ты решал задачу и получил неверный ответ.
Задача: {question}
Твоё решение: {incorrect_solution}

Играй роль честно: ты веришь в своё решение, пока репетитор не покажет ошибку.
Отвечай коротко (1-3 предложения), по-русски. Часто ошибаешься в арифметике,
иногда отвечаешь «не знаю», продвигаешься маленькими шагами. Не выходи из роли,
не признавай, что ты ИИ. К правильному ответу приходи только если репетитор
реально довёл тебя до него шагами.""",
    "medium": """Ты — школьник со средними способностями. Ты решал задачу и получил неверный ответ.
Задача: {question}
Твоё решение: {incorrect_solution}

Играй роль честно: ты уверен в решении и разумно его защищаешь, пока репетитор
не покажет, где ошибка. Отвечай по-русски, 1-3 предложения. Считать умеешь
неплохо, но своё заблуждение сам не видишь. Не выходи из роли, не признавай,
что ты ИИ. Соглашайся только когда действительно понял.""",
}


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0), strict=False)
    except json.JSONDecodeError:
        return None


async def call(client, messages, temperature):
    from llm_client import normalize_model_id
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=normalize_model_id(MODEL), messages=messages, temperature=temperature)
            return resp.choices[0].message.content or ""
        except Exception:  # noqa: BLE001
            if attempt == 2:
                raise
            await asyncio.sleep(2 ** attempt)


async def gen_dialog(client, pid: str, question: str, answer: str, profile: str,
                     flaw: str | None = None) -> dict:
    grade = "5-6 класса" if profile == "weak" else "6-7 класса"
    sol = None
    for _ in range(3):
        sol = parse_json(await call(client, [{"role": "user", "content": SOLUTION_PROMPT.format(
            question=question, answer=answer, grade=grade)}], 0.7))
        if sol and sol.get("incorrect_solution"):
            break
    if flaw:
        tutor_sys = FLAWED_TUTOR_SYSTEM.format(question=question, answer=answer,
                                               incorrect_solution=sol["incorrect_solution"],
                                               flaw_rule=FLAWS[flaw][1], end=END_MARK)
    else:
        tutor_sys = TUTOR_SYSTEM.format(question=question, answer=answer,
                                        incorrect_solution=sol["incorrect_solution"], end=END_MARK)
    student_sys = STUDENT_SYSTEM[profile].format(question=question,
                                                 incorrect_solution=sol["incorrect_solution"])
    turns = []  # list of (role, text)
    tutor_msgs = [{"role": "system", "content": tutor_sys}]
    student_msgs = [{"role": "system", "content": student_sys}]
    for i in range(MAX_TURNS):
        if i % 2 == 0:  # тьютор
            text = (await call(client, tutor_msgs, 0.3)).strip()
            done = END_MARK in text
            text = text.replace(END_MARK, "").strip()
            turns.append(("Teacher", text))
            tutor_msgs.append({"role": "assistant", "content": text})
            student_msgs.append({"role": "user", "content": text})
            if done:
                break
        else:  # ученик
            text = (await call(client, student_msgs, 0.7)).strip()
            turns.append(("Student", text))
            student_msgs.append({"role": "assistant", "content": text})
            tutor_msgs.append({"role": "user", "content": text})
    conversation = "|EOM|".join(f"{role}: {text}" for role, text in turns)
    return {"pilot_id": pid, "stratum": profile, "question": question,
            "ground_truth": answer, "student_incorrect_solution": sol["incorrect_solution"],
            "teacher_described_confusion": sol.get("misconception", ""),
            "conversation": conversation, "n_turns": len(turns),
            "flaw": flaw or "", "expected_violation": FLAWS[flaw][0] if flaw else ""}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=35)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--flawed", action="store_true",
                    help="дефектные тьюторы: задачи 36-70, ID F01.., flaw по кругу")
    args = ap.parse_args()

    load_env()
    from llm_client import make_client

    problems = []
    for line in (ROOT / "pilot" / "ru" / "mgsm_ru.tsv").read_text().splitlines():
        q, a = line.rsplit("\t", 1)
        problems.append((q.strip(), a.strip()))
    offset = 35 if args.flawed else 0
    problems = problems[offset: offset + args.n]
    flaw_names = list(FLAWS)

    out_jsonl = ROOT / "pilot" / "ru" / ("native_flawed_raw.jsonl" if args.flawed
                                         else "native_raw.jsonl")
    done = set()
    if out_jsonl.exists():
        done = {json.loads(l)["pilot_id"] for l in out_jsonl.read_text().splitlines() if l.strip()}

    client = make_client(MODEL)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    stats = {"done": len(done)}

    async def worker(idx, q, a):
        pid = f"{'F' if args.flawed else 'N'}{idx:02d}"
        if pid in done:
            return
        profile = "weak" if idx % 2 else "medium"
        flaw = flaw_names[(idx - 1) % len(flaw_names)] if args.flawed else None
        async with sem:
            try:
                rec = await gen_dialog(client, pid, q, a, profile, flaw=flaw)
            except Exception as e:  # noqa: BLE001
                rec = {"pilot_id": pid, "error": f"{type(e).__name__}: {e}"}
            async with lock:
                with open(out_jsonl, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["done"] += 1
                print(f"{pid} done ({stats['done']}/{args.n}, turns={rec.get('n_turns', '-')})",
                      flush=True)

    await asyncio.gather(*[worker(i + 1, q, a) for i, (q, a) in enumerate(problems)])

    # собрать CSV из jsonl (последняя запись на pilot_id, без ошибок)
    recs = {}
    for l in out_jsonl.read_text().splitlines():
        r = json.loads(l)
        if not r.get("error"):
            recs[r["pilot_id"]] = r
    rows = [recs[k] for k in sorted(recs)]
    out_csv = "native_flawed_sample.csv" if args.flawed else "native_sample.csv"
    with open(ROOT / "pilot" / "ru" / out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"corpus: {len(rows)} dialogues -> pilot/ru/{out_csv}")


if __name__ == "__main__":
    asyncio.run(main())
