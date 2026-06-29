import argparse
import pandas as pd
import ast

parser = argparse.ArgumentParser()
parser.add_argument("--parquet", required=True)
args = parser.parse_args()

df = pd.read_parquet(args.parquet)

env_content = '''import requests

def wikipedia_search(query: str, top_n: int = 5):
    url = "<search-url-placeholder>/search"
    
    if query == '':
        return 'invalid query'
    
    data = {'query': query, 'top_n': top_n}
    response = requests.post(url, json=data)
    retrieval_text = ''
    for line in response.json():
        retrieval_text += f"{line['contents']}\\n\\n"
    retrieval_text = retrieval_text.strip()
    
    return retrieval_text'''

func_schemas_content = '''[
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Search Wikipedia for a given query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query to search for."
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of results to return. The default value is 5.",
                        "default": 5
                    }
                },
                "required": [
                    "query"
                ]
            }
        }
    }
]'''

def add_env_info(extra_info):
    if isinstance(extra_info, str):
        try:
            extra_info = ast.literal_eval(extra_info)
        except:
            extra_info = eval(extra_info)
    
    extra_info['env'] = env_content
    extra_info['func_schemas'] = func_schemas_content
    return extra_info

df['extra_info'] = df['extra_info'].apply(add_env_info)

df.to_parquet(args.parquet, index=False)

print("文件已更新完成！")
print("第一行更新后的extra_info包含的键：", list(df['extra_info'].iloc[0].keys()))
