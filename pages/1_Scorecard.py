import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Scorecard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# helpers/wrapper for saving generated schemas
def save_schemas(results: dict, ticket_id: str | None = None):
    saved = []
    failed = []

    items = (
        [(ticket_id, results[ticket_id])]
        if ticket_id
        else results.items()
    )

    for tid, data in items:
        pipeline_instance = data["pipeline"]
        try:
            pipeline_instance.save_rendered_schema(save_to_bq=False)
            saved.append(tid)
        except Exception as e:
            failed.append((tid, str(e)))
    
    if saved:
        if ticket_id:
            st.success(f"Schemas for ticket {ticket_id} saved successfully!")
        else:
            st.succes(f"Schemas saved for tickets: {', '.join(saved)}")

    if failed:
        st.error("Some schemas failed to save:")
        for tid, err in failed:
            st.error(f"- {tid}: {err}")

if "analysis_results" not in st.session_state:
    st.warning("No result available yet. Please run an analysis on the main form page.")
else:
    results = st.session_state["analysis_results"]

    st.header("Intent Analysis Summary Results")

    ticket_ids = list(results.keys())
    selected_ticket = st.selectbox("Select a Ticket ID:", ticket_ids)

    if selected_ticket:
        data = results[selected_ticket]
        result = data["result"]
        generated_schema = data["pipeline"].rendered

        st.subheader(f"Ticket ID: {selected_ticket}")

        st.metric(
            label="Top Intent",
            value=result["top_intent"],
            delta=f"{result["top_confidence"] * 100:.0f}% confidence"
        )

        st.subheader("Intent Scorecard")
        score_df = pd.DataFrame(result["scorecard"])
        st.bar_chart(score_df.set_index("intent"))

        st.subheader("Rationale")
        st.info(result["rationale"])

        st.subheader("Evidence")
        with st.expander("See supporting evidence"):
            for item in result["evidence"]:
                st.markdown(f"- {item}")

        st.subheader("Generated Schema")
        st.code(generated_schema, language="python")

        st.subheader("Model Info")
        st.write(f"**Model Used:** {result["model"]}")

        with st.expander("Token Usage Details"):
            st.json(result["token_usage"])

        with st.expander("Full Raw Result"):
            st.json(result)

    st.markdown("---")

    if st.button(f"Save schema for {selected_ticket}", key=f"save_{selected_ticket}"):
        save_schemas(results, ticket_id=selected_ticket)

    if st.button("Save schemas for all tickets", key="save_all"):
        save_schemas(results)