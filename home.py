from config import RUBRIC_PROMPT, INTENT_RUBRIC
from app import APP_DESC

from components import ConversationExtractor, ConversationPipeline
from clients import OpenAIClient, BigQueryClient
import streamlit as st
import asyncio

async def validate_api_key(api_key: str) -> bool:
    try:
        openai = await OpenAIClient(api_key).init_async_client()
        await openai.models.list()
        return True
    except Exception:
        return False

async def main(ticket_id: str, api_key: str):
    openai = await OpenAIClient(api_key).init_async_client()
    bq = BigQueryClient()

    extractor = ConversationExtractor(bq, ticket_id)
    raw = extractor.get_convo_str()
    parsed = extractor.parse_conversation(raw)
    stats = extractor.convo_stats(parsed)

    pipeline = ConversationPipeline(
        openai_client=openai,
        conversation=parsed,
        conversation_stats=stats,
        rubric=RUBRIC_PROMPT,
        intent_prompt=INTENT_RUBRIC
    )
    return await pipeline.run()

async def home():
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="openai_api_key", type="password")

    with st.container(border=True):
        st.title("Chat Analysis: Intent Evaluation Dashboard")
        st.caption(APP_DESC)

    with st.container():
        ticket_id = st.text_input("Ticket ID:")

        if st.button("Analyze"):
            if not openai_api_key:
                st.error("Please provide a valid OpenAI API key!")
                return

            with st.spinner("Validating API key..."):
                is_valid = await validate_api_key(openai_api_key)
            if not is_valid:
                st.error("Invalid API key.")
                return

            placeholder = st.empty()
            with placeholder, st.spinner("Analyzing ticket..."):
                results = await main(ticket_id, openai_api_key)

            placeholder.write(results)


if __name__ == "__main__":
    asyncio.run(home())