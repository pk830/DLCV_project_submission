conda create -n gnn_ovsgg python=3.10
conda activate gnn_ovsgg
pip install torch torchvision transformers torch_geometric pillow scipy wandb mmatplotlib

#Download the dataset
Download all files(roughly 20GB) from "https://cs.stanford.edu/people/dorarad/gqa/download.html"
You can create 4 folders namely "closed_vocab", "novel_obj", "novel_relationships", "novel_obj_and_relationships"
Run the corresponding files namesly "closed_vocab.py", "novel_obj.py", "rel.py", "obj_rel.py"

#Training
#We perform a small hack to avoid huge memory requirements while training the whole pipeline by caching the VLM hallucinated triplets, Detected Entities, Bounding Boxes.
run - python cache_vlm.py 
# Stores the output in: 
#closed_vocab: "/home/venky/koushikpavan/gnnovsgg_new_april/vlm_cache merged_cache_gqa_closed_vocab_train.json"
#similarly for other categories.
run - python train.py

#Inference:
if you would like to check the output given an image path:
1. Modify image_path in lightweight_pair_vlm.py
2. run python lightweight_pair_vlm.py

if you want to check after your pre-training:
python inference.py

You would need atleast 48GB VRAM to run cache_vlm.py because it loads Grounding-DINO, QWEN-Vl 7B, SimCSE