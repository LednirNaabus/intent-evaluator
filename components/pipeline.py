from components import (
    ConversationExtractor,
    SchemaAwareExtractor,
    ConvoExtractSchema,
    ConvoDataExtract,
    IntentEvaluator
)
from clients import OpenAIClient, BigQueryClient
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
        convo_data_extractor: ConvoDataExtract
    ):
        self.openai_client = openai_client
        self.convo_data_extractor = convo_data_extractor