import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import pandas as pd

if "analysis_results" not in st.session_state:
    st.warning("No result available yet. Please run an analysis on the main form page.")
else:
    summary = st.session_state["analysis_results"]
    per_class = summary["per_class_accuracy"]

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

    st.divider()

    df2 = pd.DataFrame(per_class)
    df2.index = [f"Iteration {i+1}" for i in range(len(df2))]
    st.subheader("Per-Intent Accuracy over iterations")
    st.line_chart(df2)

    st.divider()

    cm = summary["confusion_matrix"]
    labels = summary["labels"]

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    st.subheader("Per-Intent Accuracy")
    df3 = pd.DataFrame(summary["per_class_accuracy"])
    df3.index = [f"Iteration {i+i}" for i in range(len(df3))]
    st.dataframe(df3.style.format("{:.2%}"))