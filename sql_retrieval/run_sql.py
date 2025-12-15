from config.DB_Connection import get_connection

def run_sql_query(sql: str):
    # safety: allow only SELECT queries
    cleaned = sql.strip().lower()
    if not cleaned.startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)   # return dict results

    cursor.execute(sql)
    rows = cursor.fetchall()

    cursor.close()
    conn.close()
    return rows