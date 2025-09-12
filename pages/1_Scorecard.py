import streamlit as st
import pandas as pd

if "analysis_results" not in st.session_state:
    st.warning("No result available yet. Please run an analysis on the main form page.")
else:
    st.write("Ticket ID:", st.session_state.get("ticket_id", "N/A"))
    results = st.session_state["analysis_results"]

    st.header("Intent Analysis Summary Results")
    st.metric(
        label="Top Intent",
        value=results["top_intent"],
        delta=f"{results["top_confidence"] * 100:.0f}% confidence"
    )

    st.subheader("Intent Scorecard")
    score_df = pd.DataFrame(results["scorecard"])
    st.bar_chart(score_df.set_index("intent"))

    st.subheader("Rationale")
    st.info(results["rationale"])

    st.subheader("Evidence")
    with st.expander("See supporting evidence"):
        for item in results["evidence"]:
            st.markdown(f"- {item}")

    st.subheader("Model Info")
    st.write(f"**Model Used:** {results["model"]}")

    with st.expander("Token Usage Details"):
        st.json(results["token_usage"])

    with st.expander("Full Raw Result"):
        st.json(results)

    if "pipeline_instance" in st.session_state:
        if st.button("Save Rendered Schema"):
            try:
                st.session_state["pipeline_instance"].save_rendered_schema()
                st.success("Schema saved successfully!")
            except Exception as e:
                st.error(f"Failed to save schema: {e}")