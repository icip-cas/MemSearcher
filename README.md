# MemSearcher
Code release for paper "[MemSearcher: Training LLMs to Reason, Search and Manage Memory via End-to-End Reinforcement Learning](https://arxiv.org/abs/2511.00805)"

## 📷 Abstract
Typical search agents concatenate the entire interaction history into the LLM context, preserving information integrity but producing long, noisy contexts, resulting in high computation and memory costs. In contrast, using only the current turn avoids this overhead but discards essential information. This trade-off limits the scalability of search agents. To address this challenge, we propose MemSearcher, an agent workflow that iteratively maintains a compact memory and combines the current turn with it. At each turn, MemSearcher fuses the user's question with the memory to generate reasoning traces, perform search actions, and update memory to retain only information essential for solving the task. This design stabilizes context length across multi-turn interactions, improving efficiency without sacrificing accuracy. To optimize this workflow, we introduce multi-context GRPO, an end-to-end RL framework that jointly optimize reasoning, search strategies, and memory management of MemSearcher Agents. Specifically, multi-context GRPO samples groups of trajectories under different contexts and propagates trajectory-level advantages across all conversations within them. Trained on the same dataset as Search-R1, MemSearcher achieves significant improvements over strong baselines on seven public benchmarks: +11% on Qwen2.5-3B-Instruct and +12% on Qwen2.5-7B-Instruct relative average gains. Notably, the 3B-based MemSearcher even outperforms 7B-based baselines, demonstrating that striking a balance between information integrity and efficiency yields both higher accuracy and lower computational overhead.

## To-do List
- [x] Release paper
- [ ] Release our MemSearcher models
- [ ] Release code

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
