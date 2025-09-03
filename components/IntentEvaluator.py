from components.schemas import IntentEvaluation
from config import INTENT_EVALUATOR_PROMPT
from clients import OpenAIClient

from typing import Dict, List, Any, Optional
from textwrap import dedent
import json

# helper function to derive intent prompt
def user_msg(rubric_text: str, signals: Dict[str, Any], require_evidence: bool):
    prompt = f"""
        INTENT RATING RUBRIC (authoritative):
    <<<
    {rubric_text}
    >>>

    SIGNALS (combined extracted + computed conversation data):
    <<<JSON
    {json.dumps(signals, ensure_ascii=False)}
    >>>

    TASK:
    1) Derive the full intent set from the rubric (preserve the intent text and order).
    2) Assign a confidence score in [0,1] to EACH intent; scorecard should sum to 1.0.
    3) Choose top_intent = argmax(scorecard) and top_confidence = scorecard[top_intent].
    4) Provide a short rationale and { '2–5' if require_evidence else '0–5' } evidence items
       (citing concrete fields/timestamps/snippets from SIGNALS).

    OUTPUT:
    Return a single JSON object with EXACTLY these keys and shapes:

    - intents: str[]
    • At least 1 item.
    • Must list all intent levels derived from the rubric in order (lowest → highest).

    - scorecard: [intent, score][]
    • An array (min 1 item), where each item is an object:
        {{ "intent": str, "score": number in [0,1] }}
    • Include one entry per intent in `intents`.
    • Scores should sum to ~1.0 (small rounding error acceptable).
    • Do not include any extra properties in each item.

    - top_intent: str
    • Must be one of the intents in `intents`.

    - top_confidence: number in [0,1]
    • Must equal the score assigned to `top_intent`.

    - rationale: str
    • Short explanation (≤5 lines).

    - evidence: str[]
    • Zero or more concise items citing concrete fields/timestamps/snippets from SIGNALS.

    No additional keys. Respond with JSON only.
    """
    return prompt

class IntentEvaluator:
    def __init__(
        self,
        rubric_text: str,
        signals: Dict[str, Any],
        *,
        model: str = "gpt-4o-mini",
        client: Optional[OpenAIClient] = None,
        timeout_ms: int = 30_000,
        require_evidence: bool = True
    ) -> Dict[str, Any]:
        if not (rubric_text and rubric_text.strip()):
            raise ValueError("rubric_text must be non-empty.")
        if not isinstance(signals, dict) or not signals:
            raise ValueError("signals must be a non-empty dict.")
        self.rubric_text = rubric_text
        self.signals = signals
        self.model = model
        self.client = client
        self.timeout_ms = timeout_ms
        self.require_evidence = require_evidence
        self.system_message = dedent(INTENT_EVALUATOR_PROMPT)
        self.user_message = dedent(user_msg(self.rubric_text, self.signals, self.require_evidence))

    # call OpenAI Responses API
    async def call_responses_api(self) -> Any:
        response = await self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": self.user_message}
            ],
            text_format=IntentEvaluation,
            temperature=0,
            max_output_tokens=2000,
            timeout=self.timeout_ms
        )

        content_text = getattr(response, "output_text", None)
        if content_text is None:
            # fallback for older SDKs
            block = response.output[0].content[0]
            content_text = block.text

        return response, json.loads(content_text)

    async def generate_scorecard(self, response: Any, result: Dict[str, List]):
        intents = list(result.get("intents", []))
        raw_scorecard = result.get("scorecard", [])
        score_map: Dict[str, float] = {}

        if isinstance(raw_scorecard, list):
            for item in raw_scorecard:
                if not isinstance(item, dict):
                    continue
                intent = item.get("intent")
                val = item.get("score")
                if isinstance(intent, str) and isinstance(val, (int, float)):
                    score_map[intent] = float(val)
        elif isinstance(raw_scorecard, dict):
            score_map = {
                k: float(v) for k, v in raw_scorecard.items() if isinstance(v (int, float))
            }
        else:
            score_map = {}

        for intent in intents:
            score_map.setdefault(intent, 0.0)

        score_map = {k: float(v) for k, v in score_map.items() if k in intents}

        for k in list(score_map.keys()):
            v = score_map[k]
            score_map[k] = 0.0 if v < 0 else (1.0 if v > 1 else v)

        total = sum(score_map.values())
        if total <= 0:
            if intents:
                uniform = 1.0 / len(intents)
                score_map = {k: uniform for k in intents}
            else:
                score_map = {}
        else:
            score_map = {k: v / total for k, v in score_map.items()}

        top_intent = max(score_map.items(), key=lambda kv: kv[1])[0] if score_map else None
        score_map = {k: round(v, 6) for k, v in score_map.items()}

        rounded_sum = sum(score_map.values())
        delta = round(1.0 - rounded_sum, 6)
        if top_intent is not None and abs(delta) >= 1e-6:
            score_map[top_intent] = round(max(0.0, min(1.0, score_map[top_intent] + delta)), 6)

        top_confidence = score_map.get(top_intent) if top_intent is not None else None

        scorecard_list: List[Dict[str, Any]] = [
            {
                "intent": intent,
                "score": score_map.get(intent, 0.0)
            } for intent in intents
        ]

        result["scorecard"] = scorecard_list
        result["top_intent"] = top_intent
        result["top_confidence"] = top_confidence
        result["model"] = self.model
        result["scorecard"] = scorecard_list
        token_usage = response.usage.model_dump_json()
        result["token_usage"] = json.loads(token_usage)

        if self.require_evidence and not result.get("evidence"):
            result["evidence"] = []

        return result