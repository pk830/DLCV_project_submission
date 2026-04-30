"""
dataloader.py -- Dataset classes for Visual Genome and GQA.

Both datasets share the same conversation-style JSON format:

    [
      {
        "image_id": "2370799",
        "conversations": [
          {"from": "human",
           "value": "<image>\nThe following objects were detected ... 
                     {'name': 'brush', 'bbox': [215, 227, 147, 106]}, ..."},
          {"from": "gpt",
           "value": "(brush, to the right of, mud), ..."}
        ],
        "images": ["/absolute/path/to/<image_id>.jpg"]
      },
      ...
    ]

Bounding boxes in the human turn are in xywh format (x, y, width, height)
and are converted to xyxy here.

Expected directory layout on disk:

    <data_root>/
        gqa_bb_listed/
            gqa_structured_train.json
            gqa_structured_test.json
        vg_bb_listed/
            vg_structured_train.json
            vg_structured_test.json

The split ("train" / "test") selects which JSON file to load.
Image paths inside the JSON are absolute and used directly.

Parsing strategy:
    human turn  -> object detections with bounding boxes (xywh -> converted to xyxy).
    gpt   turn  -> ground-truth triplets as "(subject, relation, object)" tokens.

Each sample dict returned:
    image    : PIL.Image
    boxes    : Tensor[N, 4]  xyxy pixel coords
    labels   : List[str]     one class name per box
    triplets : List[(subj_idx, predicate_str, obj_idx)]
    image_id : str

Batch size is always 1 -- graphs have variable N/E so tensors cannot be stacked.
"""

import os
import re
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import pdb









_OBJ_RE = re.compile(
    r"\{\s*'name'\s*:\s*'([^']+)'\s*,\s*'bbox'\s*:\s*\[(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]\s*\}"
)


_TRIPLET_RE = re.compile(r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)")


def _parse_objects(human_text: str, max_objects: int):
    """
    Extract object labels and xyxy bounding boxes from the human conversation turn.

    Input bbox format is xywh (x, y, width, height); converted to xyxy here.

    Returns:
        labels : List[str]           -- class name per object (truncated to max_objects)
        boxes  : Tensor[N, 4] float  -- xyxy pixel coordinates, clamped to >= 0
    """
    matches = _OBJ_RE.findall(human_text)[:max_objects]
    if not matches:
        return [], torch.zeros(0, 4)

    labels = [m[0].strip().lower() for m in matches]
    
    boxes  = torch.tensor(
        [[max(0.0, float(m[1])),
          max(0.0, float(m[2])),
          max(0.0, float(m[1]) + float(m[3])),
          max(0.0, float(m[2]) + float(m[4]))]
         for m in matches],
        dtype=torch.float32,
    )  
    return labels, boxes


def _parse_triplets(gpt_text: str, labels: list):
    """
    Extract GT triplets from the gpt conversation turn and resolve subject/object
    strings to node indices by matching against the parsed label list.

    A triplet is kept only when both subject and object can be matched
    to a label (exact match first, then substring fallback).

    Returns List[(subj_idx, predicate_str, obj_idx)].
    DEBUG: if triplets is empty, check that gpt_text contains "(x, rel, y)" tokens.
    """
    raw = _TRIPLET_RE.findall(gpt_text)  

    def find_idx(name: str):
        """Return the first label index matching name (exact, then substring)."""
        name = name.strip().lower()
        for i, lbl in enumerate(labels):
            if lbl == name:
                return i
        
        for i, lbl in enumerate(labels):
            if name in lbl or lbl in name:
                return i
        return None

    triplets = []
    for subj_str, rel_str, obj_str in raw:
        s = find_idx(subj_str)
        o = find_idx(obj_str)
        if s is not None and o is not None and s != o:
            triplets.append((s, rel_str.strip().lower(), o))
    return triplets


def _load_entry(entry: dict, max_objects: int, transform):
    """
    Parse one JSON entry into a sample dict. Shared by all dataset classes.

    Image path is read from entry["images"][0] -- absolute path in the JSON.

    DEBUG: KeyError on "conversations" -> entry is malformed or wrong JSON format.
    DEBUG: FileNotFoundError on image open -> check the path in entry["images"][0].
    """
    image_id = str(entry["image_id"])

    img_path = entry["images"][0]  
    image    = Image.open(img_path).convert("RGB")
    if transform:
        image = transform(image)

    convs      = entry["conversations"]
    human_text = next(c["value"] for c in convs if c["from"] == "human")
    gpt_text   = next(c["value"] for c in convs if c["from"] == "gpt")

    labels, boxes = _parse_objects(human_text, max_objects)
    triplets      = _parse_triplets(gpt_text, labels) if labels else []

    return {"image": image, "boxes": boxes, "labels": labels,
            "triplets": triplets, "image_id": image_id}






class VisualGenomeDataset(Dataset):
    """
    Visual Genome dataset loaded from the conversation-style JSON format.

    Reads from: <root>/vg_bb_listed/vg_structured_train.json  (split="train")
                <root>/vg_bb_listed/vg_structured_test.json   (split="test")

    Args:
        root        : Data root directory (e.g. .../ov-sgg/data/processed).
        split       : "train" or "test".
        transform   : Optional torchvision transform applied to each image.
        max_objects : Cap on objects per image to bound graph size.
    """

    def __init__(self, root, split="train", transform=None, max_objects=20):
        self.transform   = transform
        self.max_objects = max_objects

        json_path = os.path.join(root, "vg_bb_listed", f"vg_structured_{split}.json")
        with open(json_path) as f:
            all_entries = json.load(f)

        
        self.entries = [e for e in all_entries
                        if _TRIPLET_RE.search(
                            next((c["value"] for c in e["conversations"]
                                  if c["from"] == "gpt"), "")
                        )]
        pass

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        """Parses and returns one VG sample. See _load_entry for details."""
        return _load_entry(self.entries[idx], self.max_objects, self.transform)


class GQADataset(Dataset):
    def __init__(self, root, split="train", transform=None, max_objects=20, 
                 json_path=None, cache_file=None, gqa_category="closed"):
        self.transform   = transform
        self.max_objects = max_objects

        
        category_map = {
            "closed": ("closed_vocab", "closed_vocab"),
            "novel_rel": ("novel_relationships", "novel_rels"),
            "novel_obj": ("novel_obj", "novel_objects"),
            "novel_obj_and_rel": ("novel_obj_and_rel", "novel_obj_and_rel") 
        }

        if gqa_category not in category_map:
            raise ValueError(f"Invalid gqa_category: '{gqa_category}'. Must be one of {list(category_map.keys())}")

        folder_name, file_prefix = category_map[gqa_category]

        if json_path is None:
            
            
            json_path = os.path.join(root, "gqa", folder_name, f"{file_prefix}_{split}.json")
            
        with open(json_path) as f:
            all_entries = json.load(f)

        
        self.entries = [e for e in all_entries
                        if _TRIPLET_RE.search(
                            next((c["value"] for c in e["conversations"]
                                  if c["from"] == "gpt"), "")
                        )]
        
        
        self.cache = {}
        
        if cache_file is not None and os.path.exists(cache_file):
            print(f"[INFO] GQADataset ({gqa_category}): Loading VLM Cache from {cache_file}...")
            with open(cache_file, 'r') as f:
                self.cache = json.load(f)
        elif cache_file is not None:
            print(f"[WARNING] GQADataset ({gqa_category}): Cache file {cache_file} not found. Running without cache.")
        
        entry_ids = {str(e.get("image_id")) for e in self.entries}
        cache_ids = set(self.cache.keys())
        self.valid_ids = list(entry_ids & cache_ids)
    
    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, index):
        img_id = self.valid_ids[index]
        
        vlm_data = self.cache[img_id]
        labels = vlm_data.get("labels", [])
        boxes = torch.tensor(vlm_data.get("boxes", []), dtype=torch.float32)
        cached_triplets = vlm_data.get("triplets", [])
        entry = next((e for e in self.entries if str(e.get("image_id")) == img_id), None)
        
        if entry is None:
            
            
            raise ValueError(f"Image ID {img_id} from cache not found in self.entries")

        
        img_path = entry["images"][0]
        raw_triplets = next((c["value"] for c in entry["conversations"] if c["from"] == "gpt"), "")
        import re

        def parse_triplets(triplet_str):
            """Parses '(sub, pred, obj), (sub, pred, obj)' into [(sub, pred, obj), ...]"""
            matches = re.findall(r'\(([^)]+)\)', triplet_str)
            triplets = []
            for match in matches:
                parts = [p.strip() for p in match.split(',')]
                if len(parts) == 3:
                    sub_idx, predicate, obj_idx = parts
                    triplets.append((sub_idx, predicate, obj_idx))
            return triplets

        gt_triplets = parse_triplets(raw_triplets)
        
        from PIL import Image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "triplets": gt_triplets,
            "cached_triplets": cached_triplets,
            "image_id": img_id
        }
            
class CombinedDataset(Dataset):
    def __init__(self, root, split="train", transform=None, max_objects=20, 
                 vg_cache_file=None, gqa_cache_file=None, gqa_category="closed"): 
        
        self.vg  = VisualGenomeDataset(root, split, transform, max_objects, cache_file=vg_cache_file)
        
        self.gqa = GQADataset(root, split, transform, max_objects, cache_file=gqa_cache_file, gqa_category=gqa_category)
        self._vg_len = len(self.vg)






def collate_fn(batch):
    """
    Pass-through collate -- returns the batch as a plain list of dicts.
    Standard PyTorch collation fails here because node/edge counts differ
    across samples, so each graph is handled individually in the training loop.
    """
    return batch