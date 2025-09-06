from config import OPENAI_API_KEY, RUBRIC_PROMPT, INTENT_RUBRIC
from components import (
    ConversationExtractor,
    SchemaAwareExtractor,
    ConvoExtractSchema,
    IntentEvaluator
)
from clients import OpenAIClient, BigQueryClient
from typing import List, Dict
import logging
import asyncio
import json
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class ConversationPipeline:
    """
    The main orchestrator for the schema generator using the intent criteria.
    
    The pipeline has the following flow:

    1. Build the schema
    2. Extract signals
    3. Evaluate intent
    4. Generate the score card from the intent evaluation

    Args:
        `openai_client` (OpenAIClient): An `AsyncOpenAI` client.
        `conversation` (str): The conversation string. **Note**: should be parsed.
        `conversation_stats` (List[Dict]): The stats and metadata of the parsed conversation string.
        `rubric` (str): The rubric for extracting data from the conversation.
        `intent_prompt` (str): The rubric for intent rating.
        `dir` (str): The name of the directory where the generated schemas are saved. (Default: `schemas`)

    Usage:

    ```
    async def main():
        # Set up clients and conversation data here
        pipeline = ConversationPipeline(openai_client, conversation, conversation_stats, initial_prompt)
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
        model: str = "gpt-4o-mini",
        dir: str = "schemas"
    ):
        self.openai_client = openai_client
        self.conversation = conversation
        self.conversation_stats = conversation_stats
        self.rubric = rubric
        self.intent_prompt = intent_prompt
        self.dir = dir
        self.model = model
        self.rendered = None

        os.makedirs(self.dir, exist_ok=True)

    def save_rendered_schema(self):
        if not self.rendered:
            raise ValueError("No schema built yet. Call build_schema() first.")

        existing = [
            f for f in os.listdir(self.dir)
            if f.startswith("schema_v") and f.endswith(".py")
        ]
        version_numbers = []
        for f in existing:
            try:
                num = int(f.replace("schema_v", "").replace(".py", ""))
                version_numbers.append(num)
            except ValueError:
                continue
        next_ver = max(version_numbers, default=0) + 1
        filename = os.path.join(self.dir, f"schema_v{next_ver}.py")

        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.rendered)
        
        logging.info(f"Schema saved: {filename}")

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
        return results

# Run the pipeline using asyncio.gather()
async def process_tickets(
    openai_client: OpenAIClient,
    bq_client: BigQueryClient
):
    async def run_pipeline(ticket_id):
        print(f"Ticket ID: {ticket_id}")
        convo_extractor = ConversationExtractor(bq_client, ticket_id)
        raw_convo = convo_extractor.get_convo_str()
        convo_parsed = convo_extractor.parse_conversation(raw_convo)
        convo_stats = convo_extractor.convo_stats(convo_parsed)

        pipeline = ConversationPipeline(
            openai_client=openai_client,
            conversation=convo_parsed,
            conversation_stats=convo_stats,
            rubric=RUBRIC_PROMPT,
            intent_prompt=INTENT_RUBRIC
        )
        try:
            result = await pipeline.run()
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return ticket_id, result
        except Exception as e:
            print(f"Exception occurred while running pipeline: {e}")
            return ticket_id, None

    chats = bq_client.recent_tickets(table_name="messages", limit=1)
    ticket_ids = chats["ticket_id"].to_list()
    print(f"Number of tickets: {len(ticket_ids)}")
    tasks = [run_pipeline(ticket_id) for ticket_id in ticket_ids]
    results = await asyncio.gather(*tasks)
    return {
        ticket_id: result
        for ticket_id, result in results
    }

async def main():
    openai = OpenAIClient(OPENAI_API_KEY)
    openai_client = await openai.init_async_client()
    bq_client = BigQueryClient()
    await process_tickets(openai_client, bq_client)