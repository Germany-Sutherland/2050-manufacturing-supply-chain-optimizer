# app.py
# Manufacturing Supply Chain Optimizer (Agentic AI) - Minimal MVP
# Free & open-source stack. Streamlit front-end, small ML predictor, multi-agent scoring.
# Author: Generated for you by ChatGPT

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import io
import warnings
import os

# suppress warnings for a clean demo UI
warnings.filterwarnings("ignore")

# App config
st.set_page_config(page_title="Supply Chain Agentic AI - MVP", layout="wide")

st.title("Manufacturing R&D — Supply Chain Optimizer (Agentic AI MVP)")
st.markdown(
    """
**MVP features (2–3):**
- Upload suppliers CSV (columns: supplier, location, cost_per_unit_usd, lead_time_days, reliability_percent, past_delays)
- Agentic AI: CostAgent, TimeAgent, RiskAgent → Coordinator ranks suppliers
- Predict delay risk with a tiny ML model (trained on synthetic data)
  
**Future skills shown:** multi-agent reasoning, LLM-style synthesis fallback, predictive ML.
"""
)

# ---------------------------
# Utility functions (Agents)
# ---------------------------
def cost_agent(df):
    # lower cost => higher score in [0,1]
    vals = df["cost_per_unit_usd"].astype(float).to_numpy()
    # invert and normalize
    inv = np.max(vals) - vals
    if np.ptp(inv) == 0:
        return np.ones_like(inv) * 0.5
    scores = (inv - np.min(inv)) / np.ptp(inv)
    return scores

def time_agent(df):
    # shorter lead time => higher score
    vals = df["lead_time_days"].astype(float).to_numpy()
    inv = np.max(vals) - vals
    if np.ptp(inv) == 0:
        return np.ones_like(inv) * 0.5
    scores = (inv - np.min(inv)) / np.ptp(inv)
    return scores

def risk_agent(df):
    # risk based on reliability and past_delays
    rel = df["reliability_percent"].astype(float).to_numpy()
    delays = df["past_delays"].astype(float).to_numpy()
    # higher reliability -> higher score; more past delays -> lower score
    rel_norm = (rel - rel.min()) / (rel.ptp() if rel.ptp() != 0 else 1)
    delays_inv = np.max(delays) - delays
    delays_norm = (delays_inv - delays_inv.min()) / (delays_inv.ptp() if delays_inv.ptp() != 0 else 1)
    scores = 0.6 * rel_norm + 0.4 * delays_norm
    return scores

def coordinator(df, w_cost=0.3, w_time=0.3, w_risk=0.4):
    c = df.copy().reset_index(drop=True)
    c["score_cost"] = cost_agent(c)
    c["score_time"] = time_agent(c)
    c["score_risk"] = risk_agent(c)
    c["final_score"] = w_cost * c["score_cost"] + w_time * c["score_time"] + w_risk * c["score_risk"]
    c = c.sort_values("final_score", ascending=False).reset_index(drop=True)
    return c

# ---------------------------
# Small ML predictor (synthetic train)
# ---------------------------
MODEL_PATH = "delay_predictor.joblib"

def train_synthetic_model(random_state=42):
    # create synthetic dataset to demonstrate predictive ML
    rng = np.random.default_rng(random_state)
    n = 1000
    lead = rng.integers(2, 35, size=n)  # lead time days
    cost = rng.uniform(3.0, 40.0, size=n)  # cost per unit
    reliability = rng.uniform(70, 100, size=n)  # %
    # create a synthetic "delay" label: higher lead & low reliability => delay
    score = 0.4 * (lead / lead.max()) + 0.4 * (1 - (reliability - 70) / 30) + 0.2 * (cost / cost.max())
    prob = 1 / (1 + np.exp(-12 * (score - 0.5)))  # sharpen
    labels = (rng.random(n) < prob).astype(int)
    X = np.vstack([lead, cost, reliability]).T
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    clf = RandomForestClassifier(n_estimators=50, random_state=random_state)
    clf.fit(X_train, y_train)
    joblib.dump(clf, MODEL_PATH)
    return clf

def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        try:
            clf = joblib.load(MODEL_PATH)
            return clf
        except Exception:
            pass
    # train if missing or corrupted
    return train_synthetic_model()

# ---------------------------
# LLM-style synthesis (lightweight fallback)
# ---------------------------
def synthesize_text_recommendation(top_supplier_row):
    """
    Try to use a small local model for nicer prose if available (transformers).
    Otherwise fallback to deterministic template text.
    """
    try:
        # lazy import to avoid hard dependency early
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        # use distilgpt2 which is small and available
        model_name = "distilgpt2"
        generator = pipeline("text-generation", model=model_name, tokenizer=model_name, device=-1)
        prompt = (
            f"You are an expert supply-chain strategist. Recommend why {top_supplier_row['supplier']} ("
            f"{top_supplier_row['location']}) is the top choice given cost ${top_supplier_row['cost_per_unit_usd']}, "
            f"lead time {top_supplier_row['lead_time_days']} days, reliability {top_supplier_row['reliability_percent']}%, "
            f"and predicted delay risk {top_supplier_row.get('predicted_delay_risk','N/A')}."
            " Provide 3 concise next steps."
        )
        out = generator(prompt, max_length=120, do_sample=False)
        text = out[0]["generated_text"]
        return text
    except Exception:
        # deterministic fallback
        t = (
            f"Top supplier: {top_supplier_row['supplier']} ({top_supplier_row['location']}).\n"
            f"Attributes: cost ${top_supplier_row['cost_per_unit_usd']}, lead time {top_supplier_row['lead_time_days']} days, "
            f"reliability {top_supplier_row['reliability_percent']}%.\n"
            "Predicted next steps:\n"
            "1) Place a small pilot order to verify lead time & quality.\n"
            "2) Negotiate short-term SLA for delivery performance.\n"
            "3) Monitor first shipments and re-evaluate after 1 cycle."
        )
        return t

# ---------------------------
# UI - Upload or sample
# ---------------------------
with st.sidebar:
    st.header("Quick setup")
    uploaded = st.file_uploader("Upload suppliers CSV", type=["csv"])
    use_sample = st.button("Use sample data")

if uploaded:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()
elif use_sample:
    df = pd.read_csv(io.StringIO(
        "supplier,location,cost_per_unit_usd,lead_time_days,reliability_percent,past_delays\n"
        "Alpha Metals,Germany,12.5,7,96,1\n"
        "Beta Plastics,China,3.75,21,88,4\n"
        "Gamma Electronics,Germany,25.0,10,94,0\n"
        "Delta Components,Poland,10.0,12,90,2\n"
        "EastEngineers,China,4.5,28,85,6\n"
        "LocalForge,Germany,18.0,4,99,0\n"
    ))
else:
    st.info("Upload a suppliers CSV or click 'Use sample data' in the left panel to try the app.")
    st.stop()

# Validate minimal columns
required_cols = {"supplier","cost_per_unit_usd","lead_time_days","reliability_percent","past_delays","location"}
if not required_cols.issubset(set(df.columns)):
    st.error("CSV missing required columns. Required columns:\n" + ", ".join(sorted(required_cols)))
    st.stop()

# show uploaded data
st.subheader("Supplier table (uploaded)")
st.dataframe(df.reset_index(drop=True))

# weights inputs (agent importance)
st.sidebar.header("Agent weights")
w_cost = st.sidebar.slider("Cost importance", 0.0, 1.0, 0.3, 0.05)
w_time = st.sidebar.slider("Time importance", 0.0, 1.0, 0.3, 0.05)
w_risk = st.sidebar.slider("Risk importance", 0.0, 1.0, 0.4, 0.05)
normalize = st.sidebar.checkbox("Normalize weights", value=True)
if normalize:
    s = w_cost + w_time + w_risk
    if s > 0:
        w_cost, w_time, w_risk = [v/s for v in (w_cost, w_time, w_risk)]

st.subheader("Run Agentic Analysis")
if st.button("Run Analysis"):
    # run coordinator
    ranked = coordinator(df, w_cost, w_time, w_risk)

    # load or train predictor
    clf = load_or_train_model()

    # predict delay risk for each supplier (probability)
    X = ranked[["lead_time_days","cost_per_unit_usd","reliability_percent"]].astype(float).to_numpy()
    probs = clf.predict_proba(X)[:,1]  # probability of delay
    ranked["predicted_delay_risk"] = (probs * 100).round(2)

    # show ranked table with predictions
    st.subheader("Ranked suppliers")
    show_cols = ["supplier","location","cost_per_unit_usd","lead_time_days","reliability_percent","predicted_delay_risk","final_score"]
    st.table(ranked[show_cols].head(10).round(3))

    # synthesize recommendation (LLM optional / fallback)
    top = ranked.iloc[0].to_dict()
    rec_text = synthesize_text_recommendation(top)
    st.subheader("Coordinator recommendation")
    st.write(rec_text)

    st.success("Analysis finished. You can now download the ranked table as CSV for your reports.")
    csv_bytes = ranked.to_csv(index=False).encode("utf-8")
    st.download_button("Download ranked suppliers (CSV)", data=csv_bytes, file_name="ranked_suppliers.csv", mime="text/csv")

# show notes
st.markdown("""
---
**Notes for recruiters / reviewers**

- This MVP demonstrates *agentic reasoning* (multiple agents scoring different concerns and a coordinator synthesizing a single decision).
- Delay prediction uses a lightweight RandomForest trained on synthetic data to illustrate predictive ML (can be upgraded to a real dataset).
- The app uses only open-source libraries and is ready to deploy on Streamlit Cloud from a public GitHub repo.
""")
