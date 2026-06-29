#!/bin/bash
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SAVE_PATH="${1:-${SAVE_PATH:-$REPO/corpus-large}}"
mkdir -p "$SAVE_PATH"
SAVE_PATH="$(cd "$SAVE_PATH" && pwd)"
CFG="$REPO/scripts/serving/retriever_config.yaml"
echo "[download_corpus] saving to: $SAVE_PATH"

huggingface-cli download PeterJinGo/wiki-18-e5-index part_aa part_ab \
  --repo-type dataset --local-dir "$SAVE_PATH"
huggingface-cli download PeterJinGo/wiki-18-corpus wiki-18.jsonl.gz \
  --repo-type dataset --local-dir "$SAVE_PATH"

echo "[download_corpus] assembling e5_Flat.index ..."
cat "$SAVE_PATH"/part_* > "$SAVE_PATH/e5_Flat.index"
rm -f "$SAVE_PATH"/part_aa "$SAVE_PATH"/part_ab
echo "[download_corpus] decompressing wiki-18.jsonl ..."
gzip -d -f "$SAVE_PATH/wiki-18.jsonl.gz"

echo "[download_corpus] updating $CFG ..."
sed -i "s|^index_path:.*|index_path: \"$SAVE_PATH/e5_Flat.index\"|"  "$CFG"
sed -i "s|^corpus_path:.*|corpus_path: \"$SAVE_PATH/wiki-18.jsonl\"|" "$CFG"

echo "[download_corpus] done. retriever_config.yaml now points to:"
grep -E '^(index_path|corpus_path):' "$CFG"
