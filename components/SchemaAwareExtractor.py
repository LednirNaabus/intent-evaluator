from clients import OpenAIClient
from config import SYSTEM_MSG_2

from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from textwrap import dedent
import json

class SchemaAwareExtractor:
    """
    Dynamic, schema-aware extractor for MechaniGo.ph conversations.

    - Reads an evolving intent rubric and a generated Pydantic model source.
    - Builds a prompt that tells the LLM to extract **only** fields present in `ConvoExtract`.
    - Optionally enforces the output with Structured Outputs by compiling the class and deriving JSON schema.
    """
    def __init__(
        self,
        client: OpenAIClient,
        rubric: str,
        schema_class: type[BaseModel],
        messages: List[Dict[str, str]] = [],
        model: str = "gpt-4o-mini"
    ):
        """
        Args:
            `rubric` (str): Free-text intent rubric (can evolve).
            `schema` (type[BaseModel]): Python source code string that defines `class ConvoExtract(BaseModel): ...`
            `messages` (List[Dict[str, str]]): Conversation messages as a list of `{"role": "user|assistant|agent", "content": "..."}`
            `model` (str): OpenAI model name.
            `client` (AsyncOpenAI): Optional OpenAI client.
        """
        self.model = model
        self.rubric = rubric
        self.schema_class = schema_class
        self.messages = list(messages) if messages is not None else []
        self.client = client
        
        # Lazily compiled class + schema cache
        self._compiled_ns: Optional[Dict] = None
        self._json_schema: Optional[Dict] = None
        self._chat_model_cls = None

    def build_user_prompt(self) -> str:
        """Construct a schema-aware prompt that adapts to any rubric and `ConvoExtract` source."""
        if not self.rubric:
            raise ValueError("rubric is required")
        if not self.schema_class:
            raise ValueError("schema (`ConvoExtract` source) is required")

        conversation_json = json.dumps(self.messages, ensure_ascii=False)
        return dedent(f"""
        INTENT RUBRIC (authoritative, may evolve):
        <<<
        {self.rubric}
        >>>

        CHAT ANALYSIS DATA MODEL (authoritative source code to follow verbatim):
        <<<python
        {self.schema_class.model_json_schema()}
        >>>

        CONVERSATION (array):
        <<<JSON
        {conversation_json}
        >>>

        TASK:
        1) Read the rubric and the ConvoExtract source. Treat the model’s Field descriptions as your extraction rules.
        2) Extract ONLY what the conversation supports with high confidence.
        3) Return EXACTLY ONE JSON object that conforms to ConvoExtract (no markdown, no comments).
        """).strip()

    async def extract(self) -> Dict[str, Any]:
        """
        Calls the LLM to extract a JSON object that conforms to `ConvoExtract`.

        Returns:
            `Dict`: Dictionary representing `ConvoExtract`.
        """
        messages = [
            {"role": "system", "content": dedent(SYSTEM_MSG_2).strip()},
            {"role": "user", "content": self.build_user_prompt()}
        ]

        response = await self.client.responses.parse(
            model=self.model,
            input=messages,
            text_format=self.schema_class,
            temperature=0
        )
        
        try:
            text = response.output_text
        except Exception:
            text = json.dumps(response.output[0].content[0].text)

        return json.loads(text)

    async def extract_validated(self) -> Any:
        """
        Strictly validate by instantiating the compiled `ConvoExtract` class
        with the extracted payload.

        Returns:
            A Pydantic model instance.
        """
        payload = await self.extract()
        ConvoExtract = self.schema_class
        return ConvoExtract(**payload)