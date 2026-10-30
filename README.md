# GIST: Global Inducing-Token Sparse Transformer 

Code for **"Sparse Graph Transformers with Global Inducing Tokens for Knowledge Graph Completion"**.

GIST is a sparse Graph Transformer for knowledge graph completion. For each query `(h, r, ?)` it runs a learned,
query-dependent search over the graph. At every layer, a learned priority selects the entities to expand
and the edges to propagate, and only the reached entities keep a representation. Within this explored region, GIST
combines

- **pruned relational attention** along the selected edges, and
- a small set of **global inducing tokens** that pool information from the expanded entities and broadcast it to every
  reached entity,

## Repository structure

```
train.py            training and evaluation
src/model.py        GIST: pruned relational attention, inducing tokens, search and scoring
src/metric.py       MR, MRR and Hits@k
configs/            one configuration file per dataset
data/               FB15k-237, WN18RR, NELL-995 and YAGO3-10
```

## Installation

The code was run with Python 3.10, PyTorch 2.1 (CUDA 12.1) and PyTorch Lightning 1.9.

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

## Data

**Standard benchmarks.** FB15k-237, WN18RR, NELL-995 and YAGO3-10 are included in `data/`.

**Wikidata5M.** Download the transductive split from the [Wikidata5M page](https://deepgraphlearning.github.io/project/wikidata5m), extract it, and set `RAW_DIR` in
`src/data_wd5m.py` to the directory with `wikidata5m_transductive_{train,valid,test}.txt`.

## Training

Each file in `configs/` contains the settings of one dataset:

```bash
python train.py --config configs/fb15k-237.yaml
```

Main model arguments:

| argument | meaning |
|---|---|
| `--node_ratio` | fraction of entities expanded per layer |
| `--degree_ratio` | edges propagated per layer |
| `--test_node_ratio` | node_ratio at evaluation time (default: same as training) |
| `--global_channel` | `tokens` (inducing tokens), `sum` or `relation` (global readouts), or `none` |
| `--num_tokens` | number of inducing tokens |
| `--rw_distance` | random-walk distance to the query entity (used on ogbl-wikikg2 and Wikidata5M) |
| `--neg_visited` | hard negatives per query, sampled among the reached entities |
| `--dependent` | relation vectors conditioned on the query relation |

## Evaluation

Evaluate a checkpoint on the test set:

```bash
python train.py --config configs/nell-995.yaml --eval_only --eval_ckpt <checkpoint.ckpt>
```

On ogbl-wikikg2 we follow the OGB protocol (each query is ranked against 500 fixed negatives) on random samples of
8,000 test queries. We report the mean and standard deviation over five samples (seeds 1 to 5):

```bash
python train.py --config configs/ogbl-wikikg2.yaml --devices 1 --eval_only --eval_ckpt <checkpoint.ckpt> \
  --eval_subsample 8000 --eval_seed 1
```

## Results

Test results of GIST.

| dataset | MRR | Hits@1 | Hits@10 |
|---|---:|---:|---:|
| FB15k-237 | 0.418 | 0.329 | 0.591 |
| WN18RR | 0.554 | 0.504 | 0.654 |
| NELL-995 | 0.558 | 0.490 | 0.672 |
| YAGO3-10 | 0.592 | 0.517 | 0.728 |

All experiments ran on NVIDIA A100 GPUs with 40 GB of memory.
