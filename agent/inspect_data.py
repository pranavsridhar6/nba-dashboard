# One-off: print every parquet's columns, types, row count and 2 sample rows.
import pathlib, duckdb

con = duckdb.connect()
for p in sorted(pathlib.Path("data").rglob("*.parquet")):
    src = f"read_parquet('{p.as_posix()}')"
    n = con.execute(f"SELECT COUNT(*) FROM {src}").fetchall()[0][0]
    sample = con.execute(f"SELECT * FROM {src} LIMIT 2").df()  # .df() reads all rows, so nothing stays open
    print(f"\n=== {p} ({n:,} rows)")
    for col, typ in sample.dtypes.items():
        print(f"  {col}: {typ}")
    print(sample.to_string(max_colwidth=30))