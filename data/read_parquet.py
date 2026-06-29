import argparse
import pandas as pd

# 检视一个 parquet 文件的结构。用法: python read_parquet.py --file path/to.parquet
parser = argparse.ArgumentParser()
parser.add_argument("--file", required=True, help="待检视的 parquet 路径")
df = pd.read_parquet(parser.parse_args().file)

# 基本信息
print("基本信息:")
print(f"形状: {df.shape}")
print(f"列名: {list(df.columns)}")
print(f"\n数据类型:")
print(df.dtypes)

print(f"\n缺失值:")
print(df.isnull().sum())

print(f"\n第一行数据:")
for col in df.columns:
    print(f"{col}: {df.iloc[0][col]}")
