import json
import mysql.connector
from dotenv import load_dotenv
import os
load_dotenv()

# -----------------------------
# MySQL connection config
# -----------------------------
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("MYSQL_DB")
)

cursor = db.cursor()

# -----------------------------
# Load JSON
# -----------------------------
with open("grades.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# -----------------------------
# Extract table object
# -----------------------------
table_obj = next(item for item in data if item.get("type") == "table")

table_name = table_obj["name"]
rows = table_obj["data"]

# -----------------------------
# Generate CREATE TABLE SQL
# (infer column types automatically)
# -----------------------------
sample = rows[0]

columns = []
for col, val in sample.items():
    if val is None:
        col_type = "TEXT"
    else:
        # Try to infer numeric or text
        try:
            float(val)
            col_type = "DOUBLE"
        except:
            col_type = "TEXT"
    columns.append(f"`{col}` {col_type}")

create_sql = f"CREATE TABLE IF NOT EXISTS `{table_name}` ({', '.join(columns)});"

print("Creating table...")
cursor.execute(create_sql)

# -----------------------------
# Insert all rows
# -----------------------------
cols = list(sample.keys())
col_placeholders = ", ".join(["%s"] * len(cols))
col_names = ", ".join(f"`{c}`" for c in cols)

insert_sql = f"INSERT INTO `{table_name}` ({col_names}) VALUES ({col_placeholders})"

print("Inserting rows...")
for row in rows:
    values = [row.get(c) for c in cols]
    cursor.execute(insert_sql, values)

db.commit()

print("Done! Table created and data inserted.")
cursor.close()
db.close()