import streamlit as st
import json
import os
import re

st.set_page_config(
    page_title="Rubric Issues",
    layout="wide",
    initial_sidebar_state="expanded"
)

def list_runs(base_dir: str = "rubrics/runs"):
    runs = []
    if not os.path.exists(base_dir):
        return []
    
    for entry in os.listdir(base_dir):
        date_path = os.path.join(base_dir, entry)
        try:
            subruns = os.listdir(date_path)
        except OSError:
            continue

        for sub_entry in subruns:
            if sub_entry.startswith("run"):
                runs.append(os.path.join(entry, sub_entry))

    return sorted(runs)

def list_iterations(run_dir: str):
    iterations = set()
    if not os.path.exists(run_dir):
        return []
    
    try:
        for filename in os.listdir(run_dir):
            match = re.search(r"(?:issues(?:_summary)?_v|rubric_v)(\d+)", filename)
            if match:
                iterations.add(int(match.group(1)))
    except OSError:
        return []

    return sorted(iterations)

def load_rubrics_artifacts(run_dir: str, iteration: int):
    issues = []
    summary = ""
    try:
        for filename in os.listdir(run_dir):
            if re.match(fr"issues_v{iteration}(?:_\d{{4}}-\d{{2}}-\d{{2}})?\.(json|txt)", filename):
                filepath = os.path.join(run_dir, filename)
                if filename.endswith(".json"):
                    with open(filepath, "r", encoding="utf-8") as f:
                        issues = json.load(f)
                else:
                    with open(filepath, "r", encoding="utf-8") as f:
                        issues = [line.strip() for line in f]
                break

        for filename in os.listdir(run_dir):
            if re.match(fr"issues_summary_v{iteration}(?:_\d{{4}}-\d{{2}}-\d{{2}})?\.txt", filename):
                filepath = os.path.join(run_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    summary = f.read()
                break
    except OSError:
        pass

    return issues, summary

runs = list_runs()

if not runs:
    st.warning("No rubric issues found. Run an analysis first.")
else:
    selected = st.selectbox("Select Run", runs)
    run_dir = os.path.join("rubrics/runs", selected)

    iterations = list_iterations(run_dir)

    if not iterations:
        st.warning("No iterations found in this run.")
    else:
        
        issues, summary = load_rubrics_artifacts(run_dir, iterations)

        st.subheader(f"Rubric Issues Summary")
        if summary:
            raw = json.loads(summary)
            parsed = json.dumps(raw, indent=2, ensure_ascii=False)
        else:
            parsed = "No summary available."

        st.code(
            body=parsed,
            language="json"
        )

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