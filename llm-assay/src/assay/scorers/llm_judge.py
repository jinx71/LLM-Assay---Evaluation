"""LLM-as-judge scorer for open-ended / free-text answers.

For tasks where exact strings do not capture correctness (a regulatory Q&A,
a summary), a separate judge model grades the candidate answer against a
reference on a 0–1 scale. The judge is itself an :class:`LLMProvider`, wired
in by the runner, so any backend can act as judge.

Limitation worth stating up front: LLM judges are useful but imperfect and
can be biased toward verbose or same-family answers. Treat judge scores as a
strong signal, not ground truth, and spot-check the recorded reasons.
"""

from __future__ import annotations

import json

from assay.models import CompletionRequest, Prediction, ScoreResult, TestCase
from assay.providers.base import ProviderError
from assay.scorers.base import Scorer, ScoringContext, extract_json

JUDGE_SYSTEM = (
    "You are a strict, fair grader. Compare a candidate answer to a reference "
    "answer for the same question. Respond with ONLY a JSON object of the form "
    '{"score": <float 0..1>, "reason": "<one short sentence>"} and nothing else.'
)


class LLMJudge(Scorer):
    """Grade an answer with a judge model.

    Params:
        threshold: float -- pass mark (default 0.7)
        criteria: str    -- extra grading guidance appended to the prompt
    """

    name = "llm_judge"

    async def score(
        self, test_case: TestCase, prediction: Prediction, context: ScoringContext
    ) -> ScoreResult:
        judge = context.judge_provider
        if judge is None:
            return ScoreResult(self.name, 0.0, False, "no judge provider configured")

        threshold = float(self.params.get("threshold", 0.7))
        criteria = str(self.params.get("criteria", "")).strip()

        prompt = (
            f"Question:\n{test_case.input}\n\n"
            f"Reference answer:\n{test_case.expected}\n\n"
            f"Candidate answer:\n{prediction.output}\n\n"
        )
        if criteria:
            prompt += f"Grading criteria: {criteria}\n\n"
        prompt += 'Return JSON: {"score": <0..1>, "reason": "<short>"}'

        request = CompletionRequest(
            prompt=prompt, system=JUDGE_SYSTEM, max_tokens=200, temperature=0.0
        )

        try:
            verdict: Prediction = await judge.complete(request)
        except ProviderError as exc:
            return ScoreResult(self.name, 0.0, False, f"judge error: {exc}")

        try:
            parsed = extract_json(verdict.output)
            raw_score = float(parsed["score"])
        except (ValueError, KeyError, TypeError) as exc:
            return ScoreResult(self.name, 0.0, False, f"could not parse judge verdict: {exc}")

        score = max(0.0, min(1.0, raw_score))
        reason = str(parsed.get("reason", "")) if isinstance(parsed, dict) else ""
        return ScoreResult(self.name, score, score >= threshold, reason[:200])

    @staticmethod
    def _safe_dumps(obj: object) -> str:
        try:
            return json.dumps(obj)
        except (TypeError, ValueError):
            return str(obj)
