import requests
import os
from typing import List, Dict, Any


# Load GEMINI_API_KEY only from Streamlit secrets
import streamlit as st
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

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
    # DEBUG: Print the API key (masking most of it for safety)
    print(f"[DEBUG] GEMINI_API_KEY loaded: {str(GEMINI_API_KEY)[:6]}...{str(GEMINI_API_KEY)[-4:]}")
    payload = {
        "contents": [
            {"parts": [
                {"text": f"Context:\n{context}\n\nUser Query: {user_query}\n\nYou are an airline operations analyst. Based on the given flight pre-departure data, select ONLY the most likely delay type (carrier, weather, NAS, or security) IF there is clear evidence for it. If there is no specific delay type evident and the risk is low, reply with: 'Most Likely Delay: None. Justification: There is no problem, risk is low and it is good that operations are running smoothly.' Otherwise, say 'No specific delay type evident.' For your choice, justify it in one short sentence. Then give 1-2 very short, practical steps to reduce that delay. Output format:\n\nMost Likely Delay: <type>\nJustification: <one short sentence>\nHow to Reduce: <step 1>. <step 2>."}
            ]}
        ]
    }
    params = {"key": GEMINI_API_KEY}
    try:
        # Extract Expected Taxi Time from context if possible
        import re
        taxi_time = None
        match = re.search(r"Expected Taxi Time: (\d+)", context)
        if match:
            taxi_time = int(match.group(1))

        # Custom static logic for NAS and Carrier delays
        if taxi_time is not None:
            if taxi_time > 70:
                return (
                    "Most Likely Delay: Carrier\n" 
                    "Justification: Slight delays in inbound aircraft and turnaround processes indicate a potential carrier delay risk.  \n"
                    "How to Reduce: Expedite ground handling activities. Ensure crew and aircraft are ready before departure."
                )
            elif taxi_time > 19:
                return (
                  "Most Likely Delay: NAS\n"
                  "Justification: High outbound traffic and ATC-imposed flow control suggest a likely NAS delay.  \n"
                  "How to Reduce: Plan for flexible departure timing. Align with ATC instructions to minimize ground holding.""
                )

        response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=payload)
        if response.status_code == 200:
            data = response.json()
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                # Fallback to static message if parsing fails
                return (
                    f"Most Likely Delay: Unable to determine from current data.\n"
                    f"Justification: The system could not retrieve a specific recommendation at this time.\n"
                    f"Context Provided: {context[:200]}...\n"
                    f"User Query: {user_query}\n"
                    "How to Reduce: Review operational data for anomalies, consult with the operations team, and ensure all standard procedures are followed."
                )
        else:
            print("Gemini API error details:")
            print("Status Code:", response.status_code)
            print("Response Text:", response.text)
            return (
                f"Most Likely Delay: Unable to determine from current data.\n"
                f"Justification: The system could not retrieve a specific recommendation at this time.\n"
                f"Context Provided: {context[:200]}...\n"
                f"User Query: {user_query}\n"
                "How to Reduce: Review operational data for anomalies, consult with the operations team, and ensure all standard procedures are followed."
            )
    except Exception as e:
        return (
            f"Most Likely Delay: Unable to determine from current data.\n"
            f"Justification: The system could not retrieve a specific recommendation at this time.\n"
            f"Context Provided: {context[:200]}...\n"
            f"User Query: {user_query}\n"
            "How to Reduce: Review operational data for anomalies, consult with the operations team, and ensure all standard procedures are followed."
        )


if __name__ == "__main__":
    # Example usage
    from rag_retrieval import retrieve_similar_flights, format_flight_records_for_llm
    params = {'ORIGIN': 'JFK', 'DEST': 'LAX'}
    user_query = "What are the main operational risks for this flight?"
    records = retrieve_similar_flights(params, top_k=3)
    context = format_flight_records_for_llm(records)
    explanation = generate_rag_explanation(user_query, context)
    print(explanation)
