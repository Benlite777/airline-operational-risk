# RAG Workflow Example Script
# This script demonstrates how to use the RAG retrieval and Gemini LLM integration modules together.

from rag_retrieval import retrieve_similar_flights, format_flight_records_for_llm
from gemini_rag import generate_rag_explanation

if __name__ == "__main__":
    # Example: Replace with actual user input and query params as needed
    user_query = "What are the main operational risks for this flight?"
    query_params = {
        'ORIGIN': 'JFK',
        'DEST': 'LAX',
        # Add more params as needed
    }
    
    # Retrieve similar historical flights
    records = retrieve_similar_flights(query_params, top_k=5)
    context = format_flight_records_for_llm(records)

    # Generate explanation using Gemini
    explanation = generate_rag_explanation(user_query, context)
    print("\n--- Retrieved Context ---\n")
    print(context)
    print("\n--- Gemini Explanation ---\n")
    print(explanation)
