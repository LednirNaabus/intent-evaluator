from app import APP_DESC_1, APP_DESC_2, TRAINING_INSTRUCTIONS
from config import RUBRIC_PROMPT, INTENT_RUBRIC

from clients import BigQueryClient, OpenAIClient
# from components import FeedbackLoop
from components.r_FeedbackLoop import FBL
import streamlit as st
import asyncio
import time

st.set_page_config(
    page_title="Training",
    layout="wide",
    initial_sidebar_state="expanded"
)

async def validate_api_key(api_key: str) -> bool:
    try:
        openai = await OpenAIClient(api_key).init_async_client()
        await openai.models.list()
        return True
    except Exception:
        return False

async def run_analysis(api_key: str, limit: int, rubric_prompt: str, rubric_intent: str, iterations: int):
    if not rubric_prompt or not rubric_intent:
        st.warning("No prompt found! Using default prompts...")
        rubric_prompt=RUBRIC_PROMPT
        rubric_intent=INTENT_RUBRIC

    openai = await OpenAIClient(api_key).init_async_client()
    bq = BigQueryClient()
    feedbackloop = FBL(openai, bq, limit, rubric_prompt, rubric_intent, iterations)
    results = await feedbackloop.run_feedback_loop()
    return results

def main():
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API key", key="chatgpt_api_key", type="password")

    with st.container(border=True):
        st.title("Chat Analysis: Intent Evaluation")
        st.caption(APP_DESC_1)
        st.subheader("Training Phase")
        st.caption(APP_DESC_2)
        with st.expander("Instructions"):
            st.caption(TRAINING_INSTRUCTIONS)

    with st.form("analyze_form"):
        limit = st.number_input(
            "Number of tickets:",
            min_value=1,
            max_value=100,
            value=50,
            help="Number of tickets to analyze (1-100, default: 50)"
        )
        iterations = st.number_input(
            "Number of iterations for the feedback loop:",
            min_value=1,
            max_value=10,
            value=3,
            help="Maximum number of iterations (1-10, default: 3)"
        )
        rubric_prompt = st.text_area("Your prompt:", key="rubric_prompt")
        rubric_intent = st.text_area("Your intent evaluation rubric:", key="rubric_intent")
        analyzed = st.form_submit_button("Analyze")

    if analyzed:
        if not openai_api_key:
            st.error("Please provide your OpenAI API key.")
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
        progress_bar = st.progress(0, text="Analyzing...")

        start = time.perf_counter()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            for percent in range(0, 100, 10):
                time.sleep(0.2)
                progress_bar.progress(percent, text="Analyzing...")
            
            results = loop.run_until_complete(run_analysis(
                openai_api_key, limit, rubric_prompt, rubric_intent, iterations
            ))
        finally:
            loop.close()

        elapsed = time.perf_counter() - start
        progress_bar.progress(100, text=f"Analysis completed in {elapsed:.2f} seconds.")
        placeholder.success("Analysis complete, please check the results page.")

        st.session_state["analysis_results"] = results
        
main()