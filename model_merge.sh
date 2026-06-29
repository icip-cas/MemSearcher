#!/bin/bash
set -euo pipefail
BASE_PATH=${1:?"usage: model_merge.sh <CHECKPOINT_DIR> [STEPS...]"}
shift || true
HF_CONFIG_DIR=${HF_CONFIG_DIR:-}

if [ "$#" -gt 0 ]; then
    STEPS="$*"
else
    STEPS=$(ls -d "${BASE_PATH}"/global_step_*/ 2>/dev/null | sed 's#.*global_step_##;s#/##' | sort -n)
fi

for step in ${STEPS}; do
    echo "Processing global_step_${step}..."
    HF="${BASE_PATH}/global_step_${step}/actor/huggingface"
    if [ ! -f "${HF}/config.json" ] && [ -n "${HF_CONFIG_DIR}" ]; then
        echo "  actor/huggingface missing config.json; seeding from ${HF_CONFIG_DIR}"
        mkdir -p "${HF}"
        for f in config.json generation_config.json merges.txt tokenizer_config.json tokenizer.json vocab.json; do
            [ -f "${HF_CONFIG_DIR}/$f" ] && cp "${HF_CONFIG_DIR}/$f" "${HF}/"
        done
    fi
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${BASE_PATH}/global_step_${step}/actor" \
        --target_dir "${BASE_PATH}/global_step_${step}/"
    echo "Completed global_step_${step}"
done
echo "All steps completed!"
