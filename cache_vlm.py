"""
cache_vlm.py — Offline caching script for Qwen3-VL Ground Truth relationship proposals.

Usage (Single GPU):
    python cache_vlm.py --config config.yaml --split train

Usage (Multi-GPU Sharding):
    CUDA_VISIBLE_DEVICES=0 python cache_vlm.py --config config.yaml --split train --shard 0 --num_shards 4
    CUDA_VISIBLE_DEVICES=1 python cache_vlm.py --config config.yaml --split train --shard 1 --num_shards 4
"""

import json
import yaml
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import os

from dataloader import VisualGenomeDataset, GQADataset, CombinedDataset
from vlm import QwenVLProposer

def main():
    p = argparse.ArgumentParser(description="Offline VLM Caching")
    
    
    p.add_argument("--config", default="config.yaml")
    p.add_argument(
        "--setting",
        choices=["closed_vocab", "novel_obj", "novel_relationships", "novel_obj_and_relationships", "open_vocab"],
        default=None,
        help="OV-SGG training/eval setting for GQA."
    )
    
    
    p.add_argument("--split", type=str, default="train", choices=["train", "test", "val"], help="Dataset split to cache")
    p.add_argument("--shard", type=int, default=0, help="Which chunk of the dataset to process (0-indexed)")
    p.add_argument("--num_shards", type=int, default=2, help="Total number of chunks to split the dataset into")
    p.add_argument("--save_every", type=int, default=500, help="Save cache to disk every N images")
    
    args = p.parse_args()

    
    
    
    if "WORLD_SIZE" in os.environ and "RANK" in os.environ:
        args.num_shards = int(os.environ["WORLD_SIZE"])
        args.shard = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        print(f"[INFO] torchrun detected! Auto-assigned Shard {args.shard} to {device}")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Manual mode. Using device: {device} | Shard {args.shard}/{args.num_shards}")
    
    
    
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.setting is not None:
        cfg["setting"] = args.setting

    dataset_name = cfg.get("dataset", "vg")
    data_root    = cfg.get("data_root", "./data/")
    max_obj      = cfg.get("max_objects", 20)
    setting      = cfg.get("setting", "closed_vocab")

    print(f"[INFO] Using device: {device} | Shard {args.shard + 1}/{args.num_shards}")

    
    
    
    allowed_predicates = None
    allowed_objects = None

    if dataset_name == "gqa" and setting != "open_vocab":
        
        vocab_path = "data/gqa/gqa_vocab.json" 
        print(f"[INFO] Loading vocabulary from {vocab_path} for setting: {setting} | Split: {args.split}")
        
        try:
            with open(vocab_path, "r") as f:
                vocab_data = json.load(f)
                
            train_rels = vocab_data["train"]["relationships"]
            test_rels = vocab_data["test"]["relationships"]
            train_objs = vocab_data["train"]["objects"]
            test_objs = vocab_data["test"]["objects"]
            
            
            if setting in ["closed_vocab", "novel_obj"]:
                allowed_predicates = list(set(train_rels + test_rels))
            elif setting in ["novel_relationships", "novel_obj_and_relationships"]:
                allowed_predicates = list(set(train_rels))
                
            
            
            
            if setting in ["novel_obj", "novel_obj_and_relationships"]:
                allowed_objects = list(set(train_objs)) 
                print(f"[WARNING] Novel Object Setting: Constrained to {len(allowed_objects)} BASE objects.")
            else:
                allowed_objects = list(set(train_objs + test_objs)) 
            
        except FileNotFoundError:
            print(f"\n[FATAL ERROR] Could not find {vocab_path}. Please check the path.")
            exit(1)
    else:
        print("[INFO] Open Vocabulary mode detected (or non-GQA dataset). No constraints applied.")

    
    
    
    print(f"[INFO] Loading {dataset_name.upper()} '{args.split}' split (Setting: {setting})...")
    
    if cfg.get("use_combined_dataset", False):
        dataset = CombinedDataset(data_root, args.split, max_objects=max_obj)
    elif dataset_name == "gqa":
        _FILES = {
            "train": {
                "closed_vocab":                "gqa/closed_vocab/closed_vocab_train.json",
                "novel_obj":                   "gqa/novel_obj/novel_objects_train.json",
                "novel_relationships":         "gqa/novel_relationships/novel_rels_train.json",
                "novel_obj_and_relationships": "gqa/novel_obj_and_relationships/novel_all_train.json",
            },
            "test": {
                "closed_vocab":                "gqa/closed_vocab/closed_vocab_test.json",
                "novel_obj":                   "gqa/novel_obj/novel_objects_test.json",
                "novel_relationships":         "gqa/novel_relationships/novel_rels_test.json",
                "novel_obj_and_relationships": "gqa/novel_obj_and_relationships/novel_all_test.json",
            }
        }
        split_key = "test" if args.split == "val" else args.split
        json_path = str(Path(data_root) / _FILES[split_key][setting])
        dataset = GQADataset(data_root, args.split, max_objects=max_obj, json_path=json_path)
    else:
        dataset = VisualGenomeDataset(data_root, args.split, max_objects=max_obj)

    
    
    
    total_images = len(dataset)
    chunk_size = total_images // args.num_shards
    start_idx = args.shard * chunk_size
    end_idx = total_images if args.shard == args.num_shards - 1 else start_idx + chunk_size
    
    print(f"[INFO] Dataset total: {total_images} images. This shard processing indices {start_idx} to {end_idx-1}.")

    
    
    
    print(f"[INFO] Loading Qwen3-VL onto {device}...")
    proposer = QwenVLProposer(device=device, dataset=dataset_name)

    
    shard_suffix = f"_part{args.shard}of{args.num_shards}" if args.num_shards > 1 else ""
    out_file = Path(f"qwen_cache_{dataset_name}_{setting}_{args.split}{shard_suffix}.json")
    
    cache = {}
    if out_file.exists():
        print(f"[INFO] Found existing cache file {out_file}. Resuming...")
        with open(out_file, "r") as f:
            cache = json.load(f)

    
    
    
    save_counter = 0
    
    for i in tqdm(range(start_idx, end_idx), desc=f"Caching Shard {args.shard}", position=args.shard, leave=True):
        sample = dataset[i]
        
        img_id = str(sample.get("image_id", sample.get("image_path", i)))
        
        if img_id in cache:
            continue
                    
        labels = sample["labels"]
        boxes = sample["boxes"].to(device)
        image = sample["image"]
        
        
        
        
        
        
        valid_indices = []
        if allowed_objects is not None:
            for idx, lbl in enumerate(labels):
                if lbl.lower().strip() in allowed_objects:
                    valid_indices.append(idx)
        else:
            
            valid_indices = list(range(len(labels)))
            
        
        if len(valid_indices) < 2:
            cache[img_id] = {
                "labels": labels, 
                "boxes": boxes.cpu().tolist(),
                "triplets": []
            }
            continue

        
        filtered_labels = [labels[i] for i in valid_indices]
        filtered_boxes = boxes[valid_indices]

        try:
            
            
            triplets = proposer.propose_relations(
                image=image,
                labels=filtered_labels,
                boxes=filtered_boxes,
                predicate_vocab=allowed_predicates
            )
            
            cache[img_id] = {
                "labels": filtered_labels, 
                "boxes": filtered_boxes.cpu().tolist(), 
                "triplets": triplets
            }
            
        except Exception as e:
            print(f"\n[WARNING] Failed on image index {i} (ID: {img_id}): {e}")
            cache[img_id] = {
                "labels": labels,
                "boxes": boxes.cpu().tolist(),
                "triplets": []
            }

        save_counter += 1
        if save_counter % args.save_every == 0:
            with open(out_file, "w") as f:
                json.dump(cache, f)

    print(f"\n[INFO] Finished Shard {args.shard}. Saving final JSON...")
    with open(out_file, "w") as f:
        json.dump(cache, f)
    print("[INFO] Done!")

if __name__ == "__main__":
    main()