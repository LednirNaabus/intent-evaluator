from datetime import datetime
import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Feedback Loop",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_rubric_files():
    rubric_files = []
    rubrics_dir = "rubrics/evolution"

    if os.path.exists(rubrics_dir):
        for filename in os.listdir(rubrics_dir):
            if filename.startswith("rubric_v") and filename.endswith(".txt"):
                filepath = os.path.join(rubrics_dir, filename)
                try:
                    parts = filename.replace("rubric_v", "").replace(".txt", "").split("-")
                    iteration = int(parts[0])
                    timestamp = parts[1] if len(parts) > 1 else "unknown"

                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()

                    rubric_files.append({
                        "iteration": iteration,
                        "timestamp": timestamp,
                        "filename": filename,
                        "filepath": filepath,
                        "content": content
                    })
                except (ValueError, IndexError, IOError) as e:
                    st.warning(f"Could not load file: {filename}: {e}")

    rubric_files.sort(key=lambda x: x["iteration"])
    return rubric_files

def display_comparison(rubric_files):
    if not rubric_files:
        st.info("No rubric files found. Run the analysis first.")
        return

    tab_labels = [f"Iteration {r['iteration']}" for r in rubric_files]
    tabs = st.tabs(tab_labels)

    for i, (tab, rubric_info) in enumerate(zip(tabs, rubric_files)):
        with tab:
            col1, col2 = st.columns([3,1])

            with col1:
                st.subheader(f"Rubric Version {rubric_info['iteration']}")

            with col2:
                st.text(f"Timestamp: {rubric_info['timestamp']}")
                st.text(f"File: {rubric_info['filename']}")

            st.text_area(
                "Rubric Content:",
                value=rubric_info["content"],
                height=300,
                key=f"rubric_content_{i}",
                disabled=True
            )

            char_count = len(rubric_info["content"])
            word_count = len(rubric_info["content"].split())
            st.caption(f"Characters: {char_count} | Words: {word_count}")

def display_evolution():
    rubric_files = load_rubric_files()

    if len(rubric_files) > 1:
        st.subheader("Rubric Evolution Summary")

        evolution_data = []
        for rubric_info in rubric_files:
            evolution_data.append({
                "Iteration": rubric_info['iteration'],
                "Timestamp": rubric_info['timestamp'],
                "Character Count": rubric_info['content'],
                "Word Count": len(rubric_info['content'].split()),
                "Filename": rubric_info['filename']
            })

        df = pd.DataFrame(evolution_data)
        st.dataframe(df, width='stretch', hide_index=True)

        if st.button("Show changes between iterations"):
            for i in range(1, len(rubric_files)):
                with st.expander(f"Changes from iteration {rubric_files[i-1]['iteration']} to {rubric_files[i]['iteration']}"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Previous Version:**")
                        st.text_area("", rubric_files[i-1]['content'], height=200, key=f"prev_{i}", disabled=True)

                    with col2:
                        st.write("**Current Version:**")
                        st.text_area("", rubric_files[i]['content'], height=200, key=f"curr_{i}", disabled=True)


if "analysis_results" not in st.session_state:
    st.warning("No result available yet. Please run an analysis on the main form page.")
else:
    st.header("Iterations")
    
    if hasattr(st.session_state, "feedback_loop_instance") and hasattr(st.session_state.feedback_loop_instance, "rubric_history"):
        st.subheader("Current Session Rubric History")
        rubric_history = st.session_state.feedback_loop_instance.rubric_history

        if rubric_history:
            for i, rubric_info in enumerate(rubric_history):
                with st.expander(f"Iteration {rubric_info['iteration']} - {rubric_info['timestamp']}"):
                    st.text_area(
                        "Rubric Content:",
                        value=rubric_info["content"],
                        height=250,
                        key=f"session_rubric_{i}",
                        disabled=True
                    )
        else:
            st.info("No rubric history available.")
    
    st.subheader("Intent Rubric changes over iterations")

    rubric_files = load_rubric_files()

    if rubric_files:
        display_comparison(rubric_files)
        display_evolution()
    else:
        st.info("No saved rubric files found. Run the feedback loop to generate and save rubrics.")