# from components.schemas.schemas import ConvoExtract
from components.schemas import ConvoExtract
from clients import OpenAIClient
import json

class ConvoDataExtract:
    def __init__(
        self,
        openai_client: OpenAIClient,
        ticket_id: str = None,
        prompt: str = None,
        temperature: float = 0.8
    ):
        self.openai_client = openai_client
        self.prompt = prompt
        self.ticket_id = ticket_id
        self.temperature = temperature
        self.model = "gpt-4o-mini"
        self.prompt = prompt

    async def analyze_convo(self):
        if not self.prompt:
            raise Exception("Prompt not specified.")

        messages = [
            {"role": "system", "content": self.prompt}
        ]

        try:
            response = await self.openai_client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=ConvoExtract
            )
            return {
                "data": json.loads(response.choices[0].message.content),
                "tokens": response.usage.total_tokens
            }
        except Exception as e:
            print(f"Exception occurred while analyzing conversation: {e}")
            return None