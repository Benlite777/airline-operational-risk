import sqlite3
import pandas as pd
from typing import List, Dict, Any

DB_PATH = 'flight_data.db'
TABLE_NAME = 'flights'


def retrieve_similar_flights(query_params: Dict[str, Any], top_k: int = 5) -> pd.DataFrame:
    """
    Retrieve top_k most similar flights from the database based on query_params.
    Args:
        query_params: Dictionary of column-value pairs to match (e.g., {'ORIGIN': 'JFK', 'DEST': 'LAX'})
        top_k: Number of similar records to retrieve
    Returns:
        DataFrame of similar flights
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()

    # Simple similarity: filter by exact matches on provided params
    for col, val in query_params.items():
        if col in df.columns:
            df = df[df[col] == val]

    # If more than top_k, sample top_k rows
    if len(df) > top_k:
        df = df.sample(n=top_k, random_state=42)
    return df


def format_flight_records_for_llm(records: pd.DataFrame) -> str:
    """
    Format flight records as a string for LLM context.
    """
    if records.empty:
        return "No similar historical flights found."
    return records.to_markdown(index=False)


if __name__ == "__main__":
    # Example usage
    params = {'ORIGIN': 'JFK', 'DEST': 'LAX'}
    similar = retrieve_similar_flights(params, top_k=3)
    print(format_flight_records_for_llm(similar))
