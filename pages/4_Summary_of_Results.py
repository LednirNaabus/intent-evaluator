import streamlit as st
import pandas as pd

if "analysis_results" not in st.session_state:
    st.warning("No result available yet. Please run an analysis on the main form page.")
else:
    summary = st.session_state["analysis_results"]

    st.header("Feedback Loop Summary")

    st.metric(label="Iterations", value=summary["iteration"])

    st.subheader("Initial Accuracy vs Final Accuracy")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Initial Accuracy", f"{summary['initial_accuracy']:.2%}")

    with col2:
        st.metric(
            label="Final Accuracy",
            value=f"{summary['final_accuracy']:.2%}",
            delta=f"{summary["total_improvement"]:.2%}"
        )

    st.subheader("Accuracy Progression")
    st.line_chart(summary["accuracies"])

    st.subheader("✅ Final Results")
    rows = []
    for tid, res in summary["current_results"].items():
        rows.append({
            "Ticket ID": tid,
            "Top Intent": res["result"]["top_intent"] if res["result"] else "N/A"
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True)