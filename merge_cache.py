"""
merge_cache.py — Consolidates sharded VLM cache JSONs into a single unified file.

Usage:
    python merge_cache.py --dataset gqa --setting closed_vocab --split train
    python merge_cache.py --dataset gqa --setting novel_obj --split train
"""

import os
import json
import glob
import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Merge sharded VLM cache files.")
    p.add_argument("--dataset", type=str, default="gqa", help="Dataset name")
    p.add_argument(
        "--setting", 
        type=str, 
        required=True, 
        choices=["closed_vocab", "novel_obj", "novel_relationships", "novel_obj_and_relationships", "open_vocab"],
        help="The OV-SGG setting to merge"
    )
    p.add_argument("--split", type=str, default="train", help="Dataset split (train/test/val)")
    args = p.parse_args()

    
    out_dir = Path("vlm_cache")
    out_dir.mkdir(parents=True, exist_ok=True)

    
    
    search_pattern = f"qwen_cache_{args.dataset}_{args.setting}_{args.split}_part*of*.json"
    shard_files = glob.glob(search_pattern)

    
    if not shard_files:
        search_pattern_unsharded = f"qwen_cache_{args.dataset}_{args.setting}_{args.split}.json"
        shard_files = glob.glob(search_pattern_unsharded)

    if not shard_files:
        print(f"[ERROR] No cache files found matching pattern: {search_pattern}")
        return

    print(f"[INFO] Found {len(shard_files)} shard(s) for setting '{args.setting}'. Merging...")

    
    merged_cache = {}
    total_images_skipped = 0

    for file_path in sorted(shard_files):
        print(f"  -> Loading {file_path}...")
        with open(file_path, "r") as f:
            shard_data = json.load(f)
            
            
            empty_count = sum(1 for v in shard_data.values() if not v.get("triplets", []))
            total_images_skipped += empty_count
            
            merged_cache.update(shard_data)

    
    out_file = out_dir / f"merged_cache_{args.dataset}_{args.setting}_{args.split}.json"
    
    print(f"\n[SUCCESS] Merged {len(merged_cache)} total images.")
    print(f"[STATS] Images with 0 relationships (skipped/failed): {total_images_skipped}")
    print(f"[SAVING] Writing to {out_file}...")
    
    with open(out_file, "w") as f:
        json.dump(merged_cache, f)
        
    print("[DONE]")

if __name__ == "__main__":
    main()