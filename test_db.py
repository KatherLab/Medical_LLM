import sqlite3

import pandas as pd

# Connect to the SQLite database (make sure "mimicIII.db" is in the current directory)
db_path = "/home/jeff/Downloads/mimicIII.db"
connection = sqlite3.connect(db_path)

# Create a cursor object to execute SQL queries
cursor = connection.cursor()

# Get table names
cursor.execute("SELECT name FROM sqlite_schema WHERE type='table';")
table_names = cursor.fetchall()
print([table[0] for table in table_names])

# Using WHERE
# The search condition in the WHERE has the following form:
# left_expression COMPARISON_OPERATOR right_expression
# COMPARISON_OPERATOR include --> LIKE, =, IN, BETWEEN, <=, <> or != etc
# Example:
cursor.execute(
    "SELECT DISTINCT ICD9_CODE FROM d_icd_diagnoses WHERE LONG_TITLE LIKE '%tumor%' OR LONG_TITLE LIKE '%cancer%';")
cancer_icd_codes = cursor.fetchall()
print([code[0] for code in cancer_icd_codes])

# Output tables using Pandas
print(pd.read_sql_query("SELECT * FROM cptevents LIMIT 5;", connection))

# Close the connection
connection.close()
