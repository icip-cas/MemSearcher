#!/bin/bash
MODEL_PATH=${MODEL_PATH:-/path/to/MemSearcher-nq+hotpotqa-Qwen2.5-3B-Instruct}
SERVED_NAME=${SERVED_NAME:-$(basename "$MODEL_PATH")}
TP=${TP:-1}
PORT=${PORT:-80}

python3 -m sglang.launch_server \
        --served-model-name "${SERVED_NAME}" \
        --model-path "${MODEL_PATH}" \
        --tp ${TP} \
        --mem-fraction-static 0.6 \
        --context-length 32768 \
        --enable-metrics \
        --dtype bfloat16 \
        --host 0.0.0.0 \
        --port ${PORT} \
        --trust-remote-code \
        --disable-radix-cache
