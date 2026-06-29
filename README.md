# [ACL 2026] MemSearcher

<p align="center">
📃 <a href="https://arxiv.org/abs/2511.02805" target="_blank">Paper</a> | 
🤗 <a href="https://huggingface.co/collections/yuanqianhao/memsearcher-6a3fd21445bb5fc9e78062b6">Models</a> 
</p>

MemSearcher is a search agent that keeps a **compact, iteratively-updated memory** instead of the full
interaction history, trained end-to-end with **multi-context GRPO**.

---

## Environment Setup
```bash
conda create -n memsearcher python=3.10 -y
conda activate memsearcher
pip install --upgrade pip
pip install --no-deps -r requirements.txt
pip install -e . --no-deps
pip install flash-attn --no-build-isolation
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```
The `memsearcher` environment is used for **training** (vLLM rollout), and the **retriever**.

**Model serving (evaluation) uses SGLang**, which `requirements.txt` does not include — install it in a
separate environment:
```bash
conda create -n memsearcher-sglang python=3.10 -y && conda activate memsearcher-sglang
pip install "sglang[all]==0.4.9"
```

---

## MemSearcher Weights
Please check out our [Model Zoo](https://huggingface.co/collections/yuanqianhao/memsearcher-6a3fd21445bb5fc9e78062b6) for all public MemSearcher checkpoints.

---

## Evaluation
**1. Download the corpus + prebuilt index** (wiki-18 + E5 index):
```bash
bash scripts/serving/download_corpus.sh
```

**2. Start the retriever server**:
```bash
cd scripts/serving
python retriever_serving.py --config retriever_config.yaml --num_retriever 1 --port 8000
```

**3. Serve the model with SGLang** (in the `memsearcher-sglang` env):
```bash
conda activate memsearcher-sglang
MODEL_PATH=yuanqianhao/MemSearcher-3B bash launch_server.sh
```

**4. Run the evaluation**:
```bash
SAVE_NOTE=MemSearcher-3B \
DATA_DIR=./data \
SGL_REMOTE_URL=http://127.0.0.1:80 RETRIEVER_URL=http://127.0.0.1:8000/search \
DATASETS="nq triviaqa popqa hotpotqa 2wikimultihopqa musique bamboogle" \
bash eval.sh
```

---

## Training
Train MemSearcher from a Qwen2.5-Instruct model with multi-context GRPO.

**1. Download the corpus + prebuilt index** (same artifacts as evaluation):
```bash
bash scripts/serving/download_corpus.sh
```

**2. Start the retriever server** — the rollout searches it at `SEARCH_URL`:
```bash
cd scripts/serving
python retriever_serving.py --config retriever_config.yaml --num_retriever 1 --port 8000
```

**3. Prepare the training data** — the NQ + HotpotQA training split,
re-wrapped into MemSearcher's RL format:
```bash
python data/prepare_train_data.py --data_sources nq,hotpotqa \
  --output data/nq+hotpotqa_train_converted.parquet
```

**4. Launch training**:
```bash
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
SAVE_PATH=checkpoints/MemSearcher-3B \
SEARCH_URL=http://127.0.0.1:8000 \
bash train.sh
```

**5. Merge the FSDP shards** into HuggingFace weights, then evaluate via the **Evaluation** section:
```bash
bash model_merge.sh checkpoints/MemSearcher-3B
```

---

## Citation
If you find MemSearcher useful for your research and applications, please cite using this BibTeX:
```bib
@article{yuan2025memsearcher,
  title={MemSearcher: Training LLMs to Reason, Search and Manage Memory via End-to-End Reinforcement Learning},
  author={Yuan, Qianhao and Lou, Jie and Li, Zichao and Chen, Jiawei and Lu, Yaojie and Lin, Hongyu and Sun, Le and Zhang, Debing and Han, Xianpei},
  journal={arXiv preprint arXiv:2511.02805},
  year={2025}
}
```
