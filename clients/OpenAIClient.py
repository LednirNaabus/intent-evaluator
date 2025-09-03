from openai import AsyncOpenAI, AuthenticationError, OpenAIError
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class OpenAIClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.client = None

    async def init_async_client(self) -> AsyncOpenAI:
        try:
            client = AsyncOpenAI(api_key=self.api_key, timeout=60)
            await client.models.list()
            logging.info("OpenAI client initialized successfully!")
            self.client = client
            logging.info(f"Client: {self.client}")
            return self.client
        except (AuthenticationError, OpenAIError) as e:
            logging.error(f"OpenAI client initialization failed: {e}")
            raise
        except Exception as e:
            logging.error(f"Exception occurred while initializing OpenAI client: {e}")
            raise