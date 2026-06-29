import os
import json
import datasets
import jsonlines
import argparse
import random
random.seed(42)

wikipedia_search_env = """import requests

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
    
    return retrieval_text"""

wikipedia_search_schemas = [{
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
                "required": ["query"]
            }
        }
    }
]
wikipedia_search_schemas = json.dumps(wikipedia_search_schemas, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir', 
        help='the directory of the input data, for example the dir of musique'
    )
    parser.add_argument(
        '--output_dir', 
        help='the directory of the output parquet data'
    )

    args = parser.parse_args()

    train_data_path = os.path.join(args.input_dir, 'train.jsonl')
    lines = []
    with jsonlines.open(train_data_path) as reader:
        for line in reader:
            lines.append(line)
    train_data = []
    for line in lines:
        train_data.append({
            "data_source": "musique_re_call",
            "question": line['question'],
            "ability": "re_call",
            "reward_model": {
                    "style": "rule",
                    "ground_truth": line['golden_answers']
                },
            "extra_info": {
                "id": line['id'],
                "env": wikipedia_search_env,
                "func_schemas": wikipedia_search_schemas
            }
        })

    dev_data_path = os.path.join(args.input_dir, 'dev.jsonl')
    lines = []
    with jsonlines.open(dev_data_path) as reader:
        for line in reader:
            lines.append(line)
    dev_data = []
    random.shuffle(lines)
    for line in lines[:100]:
        dev_data.append({
            "data_source": "musique_re_call",
            "question": line['question'],
            "ability": "re_call",
            "reward_model": {
                "style": "rule",
                "ground_truth": line['golden_answers']
            },
            "extra_info": {
                "id": line['id'],
                "env": wikipedia_search_env,
                "func_schemas": wikipedia_search_schemas
            }
        })

    train_dataset = datasets.Dataset.from_list(train_data)
    test_dataset = datasets.Dataset.from_list(dev_data)

    train_dataset.to_parquet(os.path.join(args.output_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.output_dir, 'test.parquet'))