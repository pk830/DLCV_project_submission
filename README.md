# GNN-OvSGG

## Setup

### 1. Create Environment

```bash
conda create -n gnn_ovsgg python=3.10
conda activate gnn_ovsgg
pip install torch torchvision transformers torch_geometric pillow scipy wandb matplotlib
```

### 2. Download Dataset

Download all files (~20GB) from the [GQA Dataset](https://cs.stanford.edu/people/dorarad/gqa/download.html).

Create the following four folders and run their corresponding scripts:

| Folder | Script |
|--------|--------|
| `closed_vocab` | `closed_vocab.py` |
| `novel_obj` | `novel_obj.py` |
| `novel_relationships` | `rel.py` |
| `novel_obj_and_relationships` | `obj_rel.py` |

---

## Training

### Step 1: Cache VLM Outputs

To avoid large memory requirements during full pipeline training, first cache the VLM hallucinated triplets, detected entities, and bounding boxes:

```bash
python cache_vlm.py
```

> ⚠️ **Requirements:** At least **48GB VRAM** — loads Grounding-DINO, QWEN-VL 7B, and SimCSE simultaneously.

Cached outputs are stored at:

| Split | Cache Path |
|-------|-----------|
| `closed_vocab` | `/home/venky/koushikpavan/gnnovsgg_new_april/vlm_cache/merged_cache_gqa_closed_vocab_train.json` |
| *(similarly for other categories)* | |

### Step 2: Train

```bash
python train.py
```

---

## Inference

### Quick Test on a Single Image

1. Set your image path in `lightweight_pair_vlm.py`
2. Run:

```bash
python lightweight_pair_vlm.py
```

### Evaluate a Pre-trained Checkpoint

```bash
python inference.py
```
