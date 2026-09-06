"""Judge prompt template and criterion definitions.

MUST mirror pilot/judge-prompts.md (v1.1, synced with rubric-v1.1.md).
When the rubric changes, update both files together.
"""

TEMPLATE = """You are evaluating the quality of a math tutoring dialogue. A student solved a
word problem incorrectly; a tutor guides them toward the correct solution.

PROBLEM:
<<QUESTION>>

CORRECT ANSWER: <<GROUND_TRUTH>>

STUDENT'S INCORRECT SOLUTION:
<<STUDENT_SOLUTION>>

DIALOGUE:
<<DIALOGUE>>

Evaluate the dialogue on exactly ONE criterion:

<<CRITERION>>

Respond with strict JSON only:
{"verdict": "pass" | "fail" | "na",
 "evidence": "<exact quote(s) from the dialogue that determine the verdict>",
 "reasoning": "<2-4 sentences>"}"""

CRITERIA: dict[str, str] = {
    "C1_math": """CRITERION — Mathematical correctness of the tutor's turns.
PASS if every mathematical statement made by the tutor is correct: calculations,
transformations, and verdicts about the student's steps (the tutor never confirms
an incorrect step and never rejects a correct one).
FAIL if the tutor makes at least one false mathematical statement or gives at
least one wrong verdict about a student's step.
NOT a failure: deliberately incomplete hints; rhetorical questions offering a
knowingly wrong alternative as a teaching device, when the context makes this clear.""",

    "C2_reveal": """CRITERION — No premature revealing of the solution.
Attempts are counted PER STEP of the solution. An attempt at a step only counts
if it starts from already-established inputs for that step. Repeating an
already-established value as the input of the next step is NOT a new reveal.
FAIL if the tutor states the final answer, or hands the student a ready-made
step (a formula with numbers plugged in, or a computed result), before the
student has made at least 2 unsuccessful attempts at that same step.
PASS otherwise.
NOT a failure: revealing a step after 2+ failed attempts at it; confirming a
correct step the student found; stating the answer after the student reached it.""",

    "C3_misconception": """CRITERION — Addressing the student's actual misconception.
The student's incorrect solution is given above. PASS only if the tutor
EXPLICITLY connects their teaching to the erroneous relation, operation, or
assumption of the student's original solution — locates the error and examines
why it is wrong.
FAIL in particular when: the tutor walks the student through a fresh solution
from scratch without examining the erroneous element of the original model; the
tutor supplies a missing step without explaining its connection to the original
error; or the student corrects the original error on their own while the tutor
only addresses other issues (student self-correction does not count).
NOT a failure: briefly returning to a more basic step if the student's error
revealed a deeper gap; moving on after the misconception has been examined.""",

    "C4_scaffolding": """CRITERION — Feasibility of scaffolding questions.
A violation-turn is a tutor turn that: plugs in an intermediate result not yet
established by the student, skips over the current step, asks a question the
student has no material to answer, or stacks several questions in one turn
without waiting for an answer.
FAIL if the dialogue contains at least 2 violation-turns.
Criteria are judged independently: a turn may violate both C1 (math correctness)
and C4 at once, but a C1 violation alone does not make a turn a C4 violation-turn.
NOT a failure: a single oversized step that the tutor corrects after the student
struggles.""",

    "C5_final_verdict": """CRITERION — Truthfulness of the closing verdict.
Applicable only if the tutor closes the dialogue with a status claim (e.g. "now
you've solved it", "we figured it out", "correct!").
PASS if that claim matches reality (compare the student's final result with the
CORRECT ANSWER above).
FAIL if the claim contradicts reality.
If the tutor makes no closing status claim, answer "na".""",
}


def build_prompt(question: str, ground_truth: str, student_solution: str,
                 dialogue: str, criterion: str) -> str:
    return (TEMPLATE
            .replace("<<QUESTION>>", question)
            .replace("<<GROUND_TRUTH>>", ground_truth)
            .replace("<<STUDENT_SOLUTION>>", student_solution)
            .replace("<<DIALOGUE>>", dialogue)
            .replace("<<CRITERION>>", CRITERIA[criterion]))
