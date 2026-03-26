import requests
import os
from typing import List, Dict, Any

# Try to load GEMINI_API_KEY from Streamlit secrets, fallback to environment variable
try:
    import streamlit as st
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
except ImportError:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent"
def generate_rag_explanation(user_query: str, context: str) -> str:
    """
    Calls Gemini API with user query and retrieved context to generate an explanation.
    Args:
        user_query: The user's question or prediction context.
        context: Retrieved historical data as a string.
    Returns:
        LLM-generated explanation string.
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"parts": [
                {"text": f"Context:\n{context}\n\nUser Query: {user_query}\n\nYou are an airline operations analyst. Based on the given flight pre-departure data, select ONLY the most likely delay type (carrier, weather, NAS, or security) IF there is clear evidence for it. If there is no specific delay type evident and the risk is low, reply with: 'Most Likely Delay: None. Justification: There is no problem, risk is low and it is good that operations are running smoothly.' Otherwise, say 'No specific delay type evident.' For your choice, justify it in one short sentence. Then give 1-2 very short, practical steps to reduce that delay. Output format:\n\nMost Likely Delay: <type>\nJustification: <one short sentence>\nHow to Reduce: <step 1>. <step 2>."}
            ]}
        ]
    }
    params = {"key": GEMINI_API_KEY}
    response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=payload)
    if response.status_code == 200:
        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return "[Error: Unexpected Gemini API response format.]"
    else:
        # Diagnostic printout for debugging
        print("Gemini API error details:")
        print("Status Code:", response.status_code)
        print("Response Text:", response.text)
        return f"[Error: Gemini API call failed. Status {response.status_code}. Details: {response.text}]"


if __name__ == "__main__":
    # Example usage
    from rag_retrieval import retrieve_similar_flights, format_flight_records_for_llm
    params = {'ORIGIN': 'JFK', 'DEST': 'LAX'}
    user_query = "What are the main operational risks for this flight?"
    records = retrieve_similar_flights(params, top_k=3)
    context = format_flight_records_for_llm(records)
    explanation = generate_rag_explanation(user_query, context)
    print(explanation)
