from components import (
    SchemaAwareExtractor,
    ConvoExtractSchema,
    IntentEvaluator
)
from clients import OpenAIClient
from typing import List, Dict
import json

class ConversationPipeline:
    """
    The main orchestrator for the schema generator using the intent criteria.

    Usage:

    ```
    async def main():
        pipeline = ConversationPipeline(api_key="YOUR_API_KEY", convo_id=convo_id, intent_prompt="YOUR_INTENT_PROMPT")
        await pipeline.run()
    ```
    """
    def __init__(
        self,
        openai_client: OpenAIClient,
        conversation: str,
        conversation_stats: List[Dict],
        rubric: str,
        intent_prompt: str,
        model: str = "gpt-4o-mini"
    ):
        self.openai_client = openai_client
        self.conversation = conversation
        self.conversation_stats = conversation_stats
        self.rubric = rubric
        self.intent_prompt = intent_prompt
        self.model = model
        self.rendered = None

    async def build_schema(self):
        schema_generator = ConvoExtractSchema(
            intent_prompt=self.rubric,
            model="gpt-4.1-mini",
            client=self.openai_client
        )
        self.schema_class = await schema_generator.build_model_class_from_source()
        parsed = await schema_generator.ask_open_ai_for_spec()
        rendered = schema_generator.render_pydantic_class(parsed)
        self.rendered = rendered

    async def extract_signals(self):
        extractor = SchemaAwareExtractor(
            rubric=self.intent_prompt,
            schema_class=self.schema_class,
            messages=self.conversation,
            client=self.openai_client,
            model=self.model
        )
        
        extracted = await extractor.extract_validated()
        return {
            "extracted_data": extracted.model_dump(),
            "stats": self.conversation_stats
        }

    async def evaluate_intent(self, signals):
        intent_eval = IntentEvaluator(
            rubric_text=self.intent_prompt,
            signals=signals,
            client=self.openai_client,
            model=self.model
        )

        response, result = await intent_eval.call_responses_api()
        score_card = await intent_eval.generate_scorecard(response, result)
        return score_card

    async def run(self):
        await self.build_schema()
        signals = await self.extract_signals()
        results = await self.evaluate_intent(signals)
        return json.dumps(results, indent=4, ensure_ascii=False)