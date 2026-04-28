import pandas as pd
df = pd.read_parquet('e:/OSFDA/data/raw/bts_flights/bts_2018_01.parquet')
print(f"Columns: {df.columns.tolist()}")
print(f"Shape: {df.shape}")
print(df.head())
