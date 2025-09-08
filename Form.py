from config import RUBRIC_PROMPT, INTENT_RUBRIC
from app import APP_DESC

from components import ConversationExtractor, ConversationPipeline
from clients import OpenAIClient, BigQueryClient
import streamlit as st
import asyncio
import time

async def validate_api_key(api_key: str) -> bool:
    try:
        openai = await OpenAIClient(api_key).init_async_client()
        await openai.models.list()
        return True
    except Exception:
        return False

async def run_analysis(api_key: str, ticket_id: str, rubric: str, intent_prompt: str):
    if not rubric or not intent_prompt:
        rubric=RUBRIC_PROMPT
        intent_prompt=INTENT_RUBRIC

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
        rubric=rubric,
        intent_prompt=intent_prompt
    )

    results = await pipeline.run()
    return results, pipeline

def main():
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatgpt_api_key", type="password")

    with st.container(border=True):
        st.title("Chat Analysis: Intent Evaluation")
        st.caption(APP_DESC)

    with st.form("analyze_form"):
        ticket_id = st.text_input("Ticket ID:")
        rubric = st.text_area("Your prompt:", key="rubric_prompt")
        intent_rubric = st.text_area("Your intent evaluation rubric:", key="rubric_intent")
        analyzed = st.form_submit_button("Analyze")

    if analyzed:
        if not openai_api_key:
            st.error("Please provide your OpenAI API key!")
            return

        with st.spinner("Validating API key..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                is_valid = loop.run_until_complete(validate_api_key(openai_api_key))
            finally:
                loop.close()

        if not is_valid:
            st.error("Invalid OpenAI API key!")
            return

        placeholder = st.empty()
        progress_bar = st.progress(0, text="Analyzing ticket...")

        start = time.perf_counter()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            for percent in range(0, 100, 10):
                time.sleep(0.2)
                progress_bar.progress(percent, text="Analyzing ticket...")

            results, pipeline = loop.run_until_complete(run_analysis(
                openai_api_key, ticket_id, rubric, intent_rubric
            ))
        finally:
            loop.close()

        elapsed = time.perf_counter() - start
        progress_bar.progress(100, text=f"Analsis completed in {elapsed:.2f} seconds.")
        placeholder.success("Analysis complete, pleaes check the Results page.")

        st.session_state["analysis_results"] = results
        st.session_state["ticket_id"] = ticket_id
        st.session_state["pipeline_instance"] = pipeline

main()