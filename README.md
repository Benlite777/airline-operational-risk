# Airline Operational Risk Dashboard

This project is an interactive dashboard and backend system for predicting and explaining operational risks in airline flights. It leverages machine learning, retrieval-augmented generation (RAG) with Gemini LLM, and a custom Streamlit UI to provide actionable insights for airline operations analysts.

## Features

- **Flight Risk Prediction:**
  - Predicts the probability of operational risk for a given flight using a trained XGBoost model.
  - Classifies risk as High, Moderate, or Low.

- **RAG-based Explanation:**
  - Uses Google Gemini LLM to generate explanations and recommendations based on flight features and historical data.
  - If risk is low, provides a unique positive message; otherwise, gives delay type, justification, and mitigation steps.

- **Custom Dashboard UI:**
  - Built with Streamlit and a custom aviation-inspired CSS theme for a modern, professional look.
  - Displays prediction results, explanations, and operational recommendations in a clear, actionable format.

- **Security:**
  - Sensitive API keys are loaded from environment variables and never committed to source control.

## How It Works

1. **User Input:**
   - The user enters flight details (origin, destination, time, carrier, etc.) via the dashboard.

2. **Prediction:**
   - The backend model predicts the probability of operational risk (e.g., delay, disruption).
   - The risk is classified as High, Moderate, or Low.

3. **RAG Explanation:**
   - The system retrieves similar historical flights and formats them as context.
   - The Gemini LLM is called with the user query and context to generate an explanation and recommendations.
   - If the risk is low, a positive message is shown; otherwise, the LLM provides delay type, justification, and mitigation steps.

4. **Display:**
   - The dashboard shows the risk level, probability, interpretation, and operational recommendation.

## File Structure

- `app.py` — Main Streamlit app, UI logic, and integration.
- `gemini_rag.py` — RAG logic, Gemini API call, and prompt engineering.
- `rag_retrieval.py` — Retrieval of similar flights and formatting for LLM.
- `xgboost_model10_with_features.joblib` — Trained XGBoost model.
- `data/` — Contains CSVs for dashboard, raw, train, and test data.
- `notebooks/` — Jupyter notebooks for data extraction, EDA, and model building.
- `.env.example` — Example environment variable file (no secrets).
- `.env` — Your real API key (not tracked by git).
- `requirements.txt` — Python dependencies.




