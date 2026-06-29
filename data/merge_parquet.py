import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--inputs", nargs="+", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

dfs = [pd.read_parquet(p) for p in args.inputs]
merged_df = pd.concat(dfs, ignore_index=True)

merged_df.to_parquet(args.output, index=False)

print("合并完成！")
for p, d in zip(args.inputs, dfs):
    print(f"{p}: {len(d)}")
print(f"合并后数据量: {len(merged_df)}")
