import pandas as pd
import sqlite3

# Load the CSV data
df = pd.read_csv('data/train_sample_2024.csv')

# Connect to SQLite database (creates file if it doesn't exist)
conn = sqlite3.connect('flight_data.db')

# Dump DataFrame to SQL table named 'flights'
df.to_sql('flights', conn, if_exists='replace', index=False)

conn.close()
print('Data dumped to flight_data.db (table: flights)')
