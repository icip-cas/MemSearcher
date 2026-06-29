import argparse
import numpy as np
import pandas as pd

ENV_CONTENT = (
    'import requests\n'
    '\n'
    'def wikipedia_search(query: str, top_n: int = 5):\n'
    '    url = "<search-url-placeholder>/search"\n'
    '    \n'
    "    if query == '':\n"
    "        return 'invalid query'\n"
    '    \n'
    "    data = {'query': query, 'top_n': top_n}\n"
    '    response = requests.post(url, json=data)\n'
    "    retrieval_text = ''\n"
    '    for line in response.json():\n'
    '        retrieval_text += f"{line[\'contents\']}\\n\\n"\n'
    '    retrieval_text = retrieval_text.strip()\n'
    '    \n'
    '    return retrieval_text'
)

FUNC_SCHEMAS_CONTENT = '''[
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


def memsearcher_row(question, golden_answers, index, split='train'):
    return {
        'data_source': 'musique_re_call',
        'question': question,
        'ability': 're_call',
        'reward_model': {
            'ground_truth': np.array(list(golden_answers), dtype=object),
            'style': 'rule',
        },
        'extra_info': {
            'env': ENV_CONTENT,
            'func_schemas': FUNC_SCHEMAS_CONTENT,
            'index': int(index),
            'split': str(split),
        },
    }


def norm_question(q):
    q = q.strip()
    if q[-1] != '?':
        q += '?'
    return q


def from_flashrag(data_sources):
    import datasets
    rows = []
    for ds in data_sources:
        d = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', ds)['train']
        for idx, ex in enumerate(d):
            rows.append(memsearcher_row(norm_question(ex['question']), ex['golden_answers'], idx))
        print(f"  + {ds}: {len(d)} rows")
    return rows


def from_parquet(paths):
    rows = []
    for p in paths:
        df = pd.read_parquet(p)
        for _, r in df.iterrows():
            ei = r['extra_info']
            rows.append(memsearcher_row(r['question'], r['golden_answers'], ei['index'], ei['split']))
        print(f"  + {p}: {len(df)} rows")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_sources', default='nq,hotpotqa')
    ap.add_argument('--searchr1_parquet', nargs='+', default=None)
    ap.add_argument('--output', default='data/nq+hotpotqa_train_converted.parquet')
    args = ap.parse_args()

    if args.searchr1_parquet:
        rows = from_parquet(args.searchr1_parquet)
    else:
        rows = from_flashrag([s for s in args.data_sources.split(',') if s])

    out = pd.DataFrame(rows)
    out.to_parquet(args.output, index=False)
    print(f"Done. {len(out)} rows -> {args.output}")


if __name__ == '__main__':
    main()
