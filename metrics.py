"""
metrics.py — Open-Vocabulary Scene Graph Generation (OVSGG) Evaluation.

Implements Recall@K and mean Recall@K (mR@K) using the unified 
FastSemanticMatcher (Alias Dictionaries) to ensure evaluation perfectly 
mirrors training-time label assignment.
"""

import torch
import torchvision.ops as tv_ops
from collections import defaultdict
from PIL import Image


from semantic_matcher import FastSemanticMatcher





OBJECT_ALIASES = {
    "boy": "person", "girl": "person", "man": "person", "woman": "person",
    "men": "person", "women": "person", "kid": "person", "child": "person",
    "guy": "person", "lady": "person", "player": "person", "people": "person",
    "sneaker": "shoe", "boot": "shoe", "sneakers": "shoe", "boots": "shoe",
    "tshirt": "shirt", "t-shirt": "shirt", "tee": "shirt",
    "cup": "glass", "mug": "glass",
    "monitor": "screen", "display": "screen",
    "specs": "glasses", "sunglasses": "glasses",
    "auto": "car", "cab": "car", "taxi": "car",
    "bicycle": "bike", "dirtbike": "bike",
    "pine": "tree", "oak": "tree",
    "road": "street",
}

class OVSGGMetrics:
    def __init__(self, novel_objects: set, novel_predicates: set, k_vals=(20, 50, 100), 
                 dataset: str = "vg"):
        self.novel_objects = {x.strip().lower() for x in novel_objects}
        self.novel_predicates = {x.strip().lower() for x in novel_predicates}
        self.k_vals = sorted(k_vals)
        
        
        self.matcher = FastSemanticMatcher(dataset=dataset)
        
        self.results = {
            "all": {"hits_at_k": {k: 0 for k in k_vals}, "total_gt": 0},
            "novel_obj": {"hits_at_k": {k: 0 for k in k_vals}, "total_gt": 0},
            "novel_rel": {"hits_at_k": {k: 0 for k in k_vals}, "total_gt": 0},
            "novel_both": {"hits_at_k": {k: 0 for k in k_vals}, "total_gt": 0},
        }
        
        self.gt_per_predicate = defaultdict(int)
        self.hits_per_predicate = {k: defaultdict(int) for k in k_vals}

    def _match_object(self, pred: str, gt: str) -> bool:
        """Maps free-form object predictions to base classes."""
        mapped_pred = OBJECT_ALIASES.get(pred, pred)
        mapped_gt = OBJECT_ALIASES.get(gt, gt)
        return mapped_pred == mapped_gt

    def _is_novel_obj(self, s_lbl: str, o_lbl: str) -> bool:
        return s_lbl in self.novel_objects or o_lbl in self.novel_objects

    def _is_novel_rel(self, rel: str) -> bool:
        return rel in self.novel_predicates

    def update(self, pred_boxes, pred_labels, pred_scores, pred_triplets, edge_scores, 
               gt_boxes, gt_labels, gt_triplets):
        
        
        scored_preds = []
        for idx, (s, rel, o) in enumerate(pred_triplets):
            joint_score = (pred_scores[s] * pred_scores[o] * edge_scores[idx]).item()
            scored_preds.append({
                "s_lbl": pred_labels[s],
                "o_lbl": pred_labels[o],
                "rel": rel,
                "s_box": pred_boxes[s].float(),
                "o_box": pred_boxes[o].float(),
                "score": joint_score
            })
            
        scored_preds.sort(key=lambda x: x["score"], reverse=True)

        gt_recalled_at_k = {k: [False] * len(gt_triplets) for k in self.k_vals}

        for gt_idx, (gt_s, gt_rel, gt_o) in enumerate(gt_triplets):
            s_lbl = gt_labels[gt_s]
            o_lbl = gt_labels[gt_o]
            rel = gt_rel
            
            novel_o = self._is_novel_obj(s_lbl, o_lbl)
            novel_r = self._is_novel_rel(rel)
            
            self.results["all"]["total_gt"] += 1
            if novel_o: self.results["novel_obj"]["total_gt"] += 1
            if novel_r: self.results["novel_rel"]["total_gt"] += 1
            if novel_o and novel_r: self.results["novel_both"]["total_gt"] += 1
            
            
            base_rel = self.matcher._normalize(rel)
            self.gt_per_predicate[base_rel] += 1

            for k in self.k_vals:
                top_k_preds = scored_preds[:k]
                
                for p in top_k_preds:
                    if gt_recalled_at_k[k][gt_idx]:
                        continue 
                        
                    
                    
                    if (self._match_object(p["s_lbl"], s_lbl) and 
                        self._match_object(p["o_lbl"], o_lbl) and 
                        self.matcher.is_match(p["rel"], rel)):
                        
                        s_iou = tv_ops.box_iou(p["s_box"].unsqueeze(0), gt_boxes[gt_s].unsqueeze(0)).item()
                        o_iou = tv_ops.box_iou(p["o_box"].unsqueeze(0), gt_boxes[gt_o].unsqueeze(0)).item()
                        
                        if s_iou >= 0.5 and o_iou >= 0.5:
                            gt_recalled_at_k[k][gt_idx] = True
                            
                            self.results["all"]["hits_at_k"][k] += 1
                            if novel_o: self.results["novel_obj"]["hits_at_k"][k] += 1
                            if novel_r: self.results["novel_rel"]["hits_at_k"][k] += 1
                            if novel_o and novel_r: self.results["novel_both"]["hits_at_k"][k] += 1
                            self.hits_per_predicate[k][base_rel] += 1
                            break

    def compute(self):
        metrics = {}
        for category, data in self.results.items():
            total = data["total_gt"]
            for k in self.k_vals:
                hits = data["hits_at_k"][k]
                metrics[f"R@{k}_{category}"] = (hits / total) if total > 0 else 0.0

        for k in self.k_vals:
            predicate_recalls = []
            for base_rel, total_gt_for_rel in self.gt_per_predicate.items():
                if total_gt_for_rel > 0:
                    hits = self.hits_per_predicate[k][base_rel]
                    predicate_recalls.append(hits / total_gt_for_rel)
            
            metrics[f"mR@{k}"] = sum(predicate_recalls) / len(predicate_recalls) if predicate_recalls else 0.0

        return metrics