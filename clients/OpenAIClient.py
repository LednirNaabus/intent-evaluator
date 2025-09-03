from openai import AsyncOpenAI, AuthenticationError, OpenAIError
import logging
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class OpenAIClient:
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls, api_key: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, api_key: str = None):
        if not self._initialized:
            self.api_key = api_key
            self.client = None
            self._initialized = True

    async def init_async_client(self) -> AsyncOpenAI:
        if self.client is not None:
            return self.client

        async with self._lock:
            if self.client is None:
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

        return self.client