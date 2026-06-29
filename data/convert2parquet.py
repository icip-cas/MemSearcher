import json
import pandas as pd
import numpy as np

def convert_hotpotqa_to_parquet(input_file, output_file):
    converted_data = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())

            question = data['question']
            golden_answers = data['golden_answers']

            converted_row = {
                'data_source': 'musique_re_call',
                'question': question,
                'ability': 're_call',
                'reward_model': {
                    'ground_truth': np.array(golden_answers, dtype=object),
                    'style': 'rule'
                },
                'extra_info': {'id': data['id']}
            }
            
            converted_data.append(converted_row)
    
    # 创建 DataFrame
    df = pd.DataFrame(converted_data)
    
    # 保存为 parquet 格式
    df.to_parquet(output_file, index=False)
    
    print(f"转换完成！共处理 {len(converted_data)} 条数据")
    print(f"输出文件：{output_file}")
    
    # 显示转换后的数据信息
    print(f"\n转换后的数据格式:")
    print(f"形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print(f"\n数据类型:")
    print(df.dtypes)
    print(f"\n第一行数据:")
    for col in df.columns:
        print(f"{col}: {df.iloc[0][col]}")

# 使用示例
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert a FlashRAG-style jsonl to a re_call training/val parquet.")
    parser.add_argument("--input_file", required=True, help="input .jsonl (fields: question, golden_answers, id)")
    parser.add_argument("--output_file", required=True, help="output .parquet path")
    args = parser.parse_args()

    convert_hotpotqa_to_parquet(args.input_file, args.output_file)
