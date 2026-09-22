
import pandas as pd
df = pd.read_parquet("C:/Users/ma1201204/Documents/Data/imagenet-1k/validation-00001-of-00014.parquet")
row = df["image"].iloc[0]
print(type(row))
print(row.keys() if isinstance(row, dict) else "raw bytes")


