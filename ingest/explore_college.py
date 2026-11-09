"""Scratch: show which column is which in the Barttorvik CSV, using two well-known players. Delete after."""
import pandas as pd

df = pd.read_parquet("data/draft/college/year=2019/college_raw.parquet")
print(f"2019 college file: {len(df):,} rows, {df.shape[1]} columns\n")
for name in ["Zion Williamson", "Ja Morant"]:
    hit = df[df.apply(lambda r: r.astype(str).str.contains(name, regex=False).any(), axis=1)]
    if hit.empty:
        print(f"{name}: not found")
        continue
    print(f"--- {name} ---")
    for col, val in hit.iloc[0].items():
        print(f"{col:>4}: {val}")
    print()