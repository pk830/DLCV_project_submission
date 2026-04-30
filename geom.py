"""
geom.py — geometric and proposal-level filtering utilities for Ov-SGG.
"""
from __future__ import annotations
import random
from typing import Callable, Dict, List, Optional, Set, Tuple
import torch
import torchvision.ops as tv_ops

Triplet = Tuple[int, str, int]
Pair = Tuple[int, int]

@torch.no_grad()
def spatially_plausible_pairs(
    boxes_xyxy: torch.Tensor,
    image_size: Tuple[int, int],
    spatial_overlap_thr: float = 0.0,
    spatial_dist_thr: float = 0.15,  
) -> Set[Pair]:
    """
    Returns ordered pairs (i, j) using Edge-to-Edge bounding box distance.
    Filters out objects that are too far apart physically, preventing background spam.
    """
    n = int(boxes_xyxy.shape[0])
    if n <= 1:
        return set()

    W, H = image_size
    diag = max((float(W) ** 2 + float(H) ** 2) ** 0.5, 1e-6)

    iou_mat = tv_ops.box_iou(boxes_xyxy.float(), boxes_xyxy.float())
    
    
    box_a = boxes_xyxy.unsqueeze(1).expand(n, n, 4)
    box_b = boxes_xyxy.unsqueeze(0).expand(n, n, 4)
    
    
    dx = torch.max(
        torch.zeros((n, n), device=boxes_xyxy.device),
        torch.max(box_a[..., 0] - box_b[..., 2], box_b[..., 0] - box_a[..., 2])
    )
    dy = torch.max(
        torch.zeros((n, n), device=boxes_xyxy.device),
        torch.max(box_a[..., 1] - box_b[..., 3], box_b[..., 1] - box_a[..., 3])
    )
    
    
    edge_dists = torch.sqrt(dx**2 + dy**2) / diag

    keep: Set[Pair] = set()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            iou_ij = float(iou_mat[i, j].item())
            d_ij = float(edge_dists[i, j].item())
            
            
            drop = (iou_ij < spatial_overlap_thr) and (d_ij > spatial_dist_thr)
            if not drop:
                keep.add((i, j))
                
    return keep




@torch.no_grad()
def inject_deterministic_spatial_relations(
    keep_pairs: Set[Pair], 
    boxes_xyxy: torch.Tensor, 
    iou_mat: torch.Tensor,
    overlap_threshold: float = 0.5
) -> List[Triplet]:
    """
    Bypasses the VLM to generate mathematically guaranteed spatial relationships.
    Uses center-point deltas to determine directionality.
    """
    spatial_triplets = []
    for s, o in keep_pairs:
        
        if iou_mat[s, o].item() > overlap_threshold:
            continue
            
        box_s, box_o = boxes_xyxy[s], boxes_xyxy[o]
        
        cx_s, cy_s = (box_s[0] + box_s[2]) / 2, (box_s[1] + box_s[3]) / 2
        cx_o, cy_o = (box_o[0] + box_o[2]) / 2, (box_o[1] + box_o[3]) / 2
        
        dx = cx_o - cx_s  
        dy = cy_o - cy_s  
        
        if abs(dx) > abs(dy):
            if dx > 0:
                spatial_triplets.append((s, "to the left of", o))
            else:
                spatial_triplets.append((s, "to the right of", o))
        else:
            if dy > 0:
                spatial_triplets.append((s, "above", o))
            else:
                spatial_triplets.append((s, "below", o))
                
    return spatial_triplets