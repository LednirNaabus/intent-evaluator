import streamlit as st
import pandas as pd

if "analysis_results" not in st.session_state:
    st.warning("No result available yet. Please run an analysis on the main form page.")
else:
    results = st.session_state["analysis_results"]

    st.header("Intent Analysis Summary Results")

    for ticket_id, data in results.items():
        result = data["result"]

        st.subheader(f"Ticket ID: {ticket_id}")

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

        st.subheader("Model Info")
        st.write(f"**Model Used:** {result["model"]}")

        with st.expander("Token Usage Details"):
            st.json(result["token_usage"])

        with st.expander("Full Raw Result"):
            st.json(result)

        if "pipeline_instance" in st.session_state:
            if st.button("Save Rendered Schema"):
                try:
                    st.session_state["pipeline_instance"].save_rendered_schema()
                    st.success("Schema saved successfully!")
                except Exception as e:
                    st.error(f"Failed to save schema: {e}")

        st.markdown("---")