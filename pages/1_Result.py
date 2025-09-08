import streamlit as st
import pandas as pd

# Simulated analysis result (replace with your real result)
result = {
    "intents": ["No Intent", "Low Intent", "Moderate Intent", "High Intent", "Hot Intent"],
    "scorecard": [
        {"intent": "No Intent", "score": 0},
        {"intent": "Low Intent", "score": 0.2},
        {"intent": "Moderate Intent", "score": 0.3},
        {"intent": "High Intent", "score": 0.4},
        {"intent": "Hot Intent", "score": 0.1}
    ],
    "top_intent": "High Intent",
    "top_confidence": 0.4,
    "rationale": "The client inquired about a PMS for a specific vehicle, providing key details but did not confirm a booking or schedule. This indicates a strong interest but not full commitment.",
    "evidence": [
        "service_category: Preventive Maintenance Services (PMS)",
        "car_brand: Nissan",
        "car_model: NV350",
        "summary: The client inquired about a PMS for an NV350 vehicle."
    ],
    "model": "gpt-4o-mini",
    "token_usage": {
        "input_tokens": 1611,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 245,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 1856
    }
}

# 🔹 Display top intent + confidence
st.header("🔍 Intent Analysis Summary")
st.metric(label="Top Intent", value=result["top_intent"], delta=f"{result['top_confidence'] * 100:.0f}% confidence")

# 🔹 Bar chart of all intents and their scores
st.subheader("📊 Intent Scorecard")
score_df = pd.DataFrame(result["scorecard"])
st.bar_chart(score_df.set_index("intent"))

# 🔹 Rationale
st.subheader("🧠 Rationale")
st.info(result["rationale"])

# 🔹 Evidence
st.subheader("📌 Evidence")
with st.expander("See supporting evidence"):
    for item in result["evidence"]:
        st.markdown(f"- {item}")

# 🔹 Metadata
st.subheader("🧾 Model Info")
st.write(f"**Model Used:** {result['model']}")

with st.expander("Token Usage Details"):
    st.json(result["token_usage"])

# 🔹 Raw JSON (optional)
with st.expander("📦 Full Raw Result"):
    st.json(result)
