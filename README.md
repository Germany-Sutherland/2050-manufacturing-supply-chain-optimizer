# 2050-manufacturing-supply-chain-optimizer
2050 Agentic AI Based Manufacturing and Supply Chain Optimization Tool 

# Supply Chain Agentic AI - Tool

Minimal Streamlit MVP demonstrating Agentic AI for Manufacturing Supply Chain optimization.
- **Features**:
  1. Upload suppliers CSV or use sample data.
  2. Agentic AI ranks suppliers (CostAgent, TimeAgent, RiskAgent) and a Coordinator synthesizes recommendation.
  3. Predicts delay risk using a small RandomForest model trained on synthetic data.

## Run locally (quick)
```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

