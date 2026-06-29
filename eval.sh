#!/bin/bash
GENERATOR_MODEL=${GENERATOR_MODEL:-/path/to/MemSearcher-nq+hotpotqa-Qwen2.5-3B-Instruct}
DATA_DIR=${DATA_DIR:-./data}
SAVE_DIR=${SAVE_DIR:-./eval}
SGL_REMOTE_URL=${SGL_REMOTE_URL:-http://127.0.0.1:80}
RETRIEVER_URL=${RETRIEVER_URL:-http://127.0.0.1:8000/search}
export SEARCH_URL=${SEARCH_URL:-${RETRIEVER_URL}}
MAX_TURNS=${MAX_TURNS:-20}
SAVE_NOTE=${SAVE_NOTE:-}
DATASETS=${DATASETS:-"nq triviaqa popqa hotpotqa 2wikimultihopqa musique bamboogle"}
HF_DATASETS_URL=https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main

EVAL_DIR="$(cd "$(dirname "$0")/scripts/evaluation" && pwd)"

download_dataset() {
    ds="$1"
    out="${DATA_DIR}/${ds}/test.jsonl"
    [ -f "${out}" ] && return 0
    case "${ds}" in
        hotpotqa|2wikimultihopqa|musique) split=dev ;;
        *) split=test ;;
    esac
    echo "===== downloading ${ds} (${split}.jsonl) ====="
    mkdir -p "${DATA_DIR}/${ds}"
    wget -q --show-progress "${HF_DATASETS_URL}/${ds}/${split}.jsonl" -O "${out}" || { echo "download ${ds} failed"; rm -f "${out}"; exit 1; }
}

for ds in ${DATASETS}; do
    download_dataset "${ds}"
    echo "===== evaluating ${ds} ====="
    python "${EVAL_DIR}/run_eval.py" \
        --config_path "${EVAL_DIR}/eval_config.yaml" \
        --method_name memsearcher \
        --data_dir "${DATA_DIR}" \
        --dataset_name "${ds}" \
        --split test \
        --save_dir "${SAVE_DIR}" \
        --save_note "${SAVE_NOTE}" \
        --sgl_remote_url "${SGL_REMOTE_URL}" \
        --remote_retriever_url "${RETRIEVER_URL}" \
        --generator_model "${GENERATOR_MODEL}" \
        --max_turns "${MAX_TURNS}"
done
