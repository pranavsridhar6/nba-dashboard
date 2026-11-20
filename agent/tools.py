from agent.db import connect

CON = connect()
MAX_ROWS= 50

BANNED = ["insert", "update", "delete", "drop", "create", "alter",
          "copy", "attach", "install", "load", "pragma",
          "read_parquet", "read_csv", "read_json"]

def check_sql(sql: str) -> str | None:
    """Return an error message if the query is unsafe, else None."""   
    # 1. strip whitespace and ONE trailing semicolon
    s = sql.strip().rstrip(";")
    low = s.lower()
    # 2. if a ';' is still inside -> multiple statements -> reject
    if ";" in s:
        return "Only one SQL statement is allowed."
    # 3. must start with 'select' or 'with' (lowercase it first)
    if not low.startswith(("select", "with")):
        return "Only SELECT queries are allowed."
    # 4. if any BANNED word appears -> reject, naming the word
    for w in BANNED:
        if w in low:
            return f"'{w}' is not allowed. Query only the provided views."
    return None


def run_sql(sql: str) -> dict:
    err =check_sql(sql)
    if err:
        return{"error": err, "sql": sql}
    try:
        df = CON.execute(sql).df()
    except Exception as e:
        return {"error": str(e), "sql": sql}
    return {
        "sql": sql,
        "row_count": len(df),
        "truncated": len(df) > MAX_ROWS,
        "columns": list(df.columns),
        "rows": df.head(MAX_ROWS).to_dict("records"),
    }


from langchain_core.tools import tool

@tool
def sql_query(sql: str) -> dict:
    """Run one read-only DuckDB SELECT against the NBA views and return
    the rows plus the exact SQL. Use this for every number you report."""
    return run_sql(sql)


if __name__ == "__main__":
    tests = [
        "SELECT team, wins, losses FROM team_season WHERE season='2025-26' ORDER BY wins DESC LIMIT 3",
        "DROP VIEW rapm",
        "SELECT 1; SELECT 2",
        "SELECT * FROM read_parquet('C:/secret.parquet')",
        "SELECT nonexistent_col FROM rapm",
    ]
    for t in tests:
        print(t, "\n ->", run_sql(t), "\n")