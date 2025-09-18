import streamlit as st
import json
import os
import re

st.set_page_config(
    page_title="Rubric Issues",
    layout="wide",
    initial_sidebar_state="expanded"
)

def list_runs(base_dir: str = "rubrics"):
    runs = []
    if os.path.exists(base_dir):
        for entry in os.listdir(base_dir):
            path = os.path.join(base_dir, entry)
            if os.path.isdir(path) and entry.startswith("run_"):
                runs.append(entry)
    return sorted(runs)

def list_iterations(run_dir: str):
    iterations = []
    for filename in os.listdir(run_dir):
        match = re.search(r"issues(?:_summary)?_v(\d+)\.(?:json|txt)", filename)
        if match:
            iterations.append(int(match.group(1)))
    return sorted(set(iterations))

def load_rubrics_artifacts(run_dir: str, iteration: int):
    issues_file = os.path.join(run_dir, f"issues_v{iteration}.json")
    summary_file = os.path.join(run_dir, f"issues_summary_v{iteration}.txt")

    issues = []
    summary = ""

    if os.path.exists(issues_file):
        with open(issues_file, "r", encoding="utf-8") as f:
            issues = json.load(f)

    if os.path.exists(summary_file):
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = f.read()

    return issues, summary

runs = list_runs()

if not runs:
    st.warning("No rubric issues found. Run an analysis first.")
else:
    selected = st.selectbox("Select Run", runs)
    run_dir = os.path.join("rubrics", selected)

    iterations = list_iterations(run_dir)

    if not iterations:
        st.warning("No iterations found in this run.")
    else:
        
        selected_iter = st.selectbox("Select Iteration", iterations)

        issues, summary = load_rubrics_artifacts(run_dir, selected_iter)

        st.subheader(f"Rubric Issues Summary")
        st.markdown(summary if summary else "_No Summary Available_")

        st.subheader("Rubric Issues (Per Ticket)")
        if issues:
            for item in issues:
                try:
                    issue_data = json.loads(item["issue"]) if isinstance(item["issue"], str) else item["issue"]
                    with st.expander(f"Ticket {item['ticket_id']}"):
                        st.write(f"**Section**: {issue_data['section']}")
                        st.write(f"**Problem**: {issue_data['problem']}")
                except Exception as e:
                    st.error(f"Failed to parse issue for ticket {item['ticket_id']}: {e}")
        else:
            st.info("No rubric issues found for selected iteration.")