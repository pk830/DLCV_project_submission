"""
features.py — Frozen CLIP (ViT-B/32) node and edge feature extraction.

Node features  [N, NODE_DIM = 1032]:
    concat( clip_visual_crop[512], clip_text_label[512], bbox_8d[8] )

Edge features  [E, EDGE_DIM = 3096]:
    concat( union_crop_visual[512], geom_relation[8],
            node_i[1032], node_j[1032], predicate_text[512] )

All CLIP weights are permanently frozen.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn
import pdb
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
NODE_DIM = 1032   
EDGE_DIM = 3096   






def _norm_box(box: torch.Tensor, W: int, H: int) -> torch.Tensor:
    """Normalise a single xyxy box to [0, 1] given image width W and height H."""
    x1, y1, x2, y2 = box.unbind(-1)
    return torch.stack([x1 / W, y1 / H, x2 / W, y2 / H])


def _bbox_8d(boxes_norm: torch.Tensor) -> torch.Tensor:
    """
    8-d geometric descriptor per normalised xyxy box.
    Output layout: (x1, y1, x2, y2, cx, cy, w, h).
    Input [N, 4] → output [N, 8].
    """
    x1, y1, x2, y2 = boxes_norm.unbind(-1)
    return torch.stack(
        [x1, y1, x2, y2, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1],
        dim=-1,
    )


def _union_box(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """Smallest xyxy box that contains both b1 and b2 (pixel coords)."""
    return torch.stack([
        torch.min(b1[0], b2[0]), torch.min(b1[1], b2[1]),
        torch.max(b1[2], b2[2]), torch.max(b1[3], b2[3]),
    ])


def _crop(image: Image.Image, box: torch.Tensor) -> Image.Image:
    """
    Crop PIL image to xyxy box, clamped to image boundaries.
    Returns the full image as a fallback for degenerate (zero-area) boxes.
    """
    x1, y1, x2, y2 = box.int().tolist()
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.width, x2), min(image.height, y2)
    if x2 <= x1 or y2 <= y1:
        return image  
    return image.crop((x1, y1, x2, y2))


def _geom_8d(bi: torch.Tensor, bj: torch.Tensor) -> torch.Tensor:
    """
    8-d relative geometry between subject box i and object box j
    (both normalised xyxy):
        (Δcx, Δcy, log(wi/wj), log(hi/hj), IoU, dist, sin θ, cos θ)

    Encodes spatial relationship in a scale-invariant way for the edge MLP.
    """
    ci = (bi[:2] + bi[2:]) / 2          
    cj = (bj[:2] + bj[2:]) / 2          
    wi, hi = bi[2] - bi[0], bi[3] - bi[1]
    wj, hj = bj[2] - bj[0], bj[3] - bj[1]
    dx, dy  = cj[0] - ci[0], cj[1] - ci[1]
    dist    = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)

    
    ix1 = torch.max(bi[0], bj[0]); iy1 = torch.max(bi[1], bj[1])
    ix2 = torch.min(bi[2], bj[2]); iy2 = torch.min(bi[3], bj[3])
    inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
    iou   = inter / (wi * hi + wj * hj - inter + 1e-6)

    return torch.stack([
        dx, dy,
        torch.log(wi / (wj + 1e-6) + 1e-6),  
        torch.log(hi / (hj + 1e-6) + 1e-6),  
        iou, dist,
        dy / (dist + 1e-6),   
        dx / (dist + 1e-6),   
    ])


class SemanticProjector(nn.Module):
    """
    Trainable MLP to project frozen CLIP text embeddings into a 
    contrastive, dataset-specific semantic space.
    """
    def __init__(self, clip_dim=512, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, clip_dim) 
        )
        
    def forward(self, x):
        return self.mlp(x)





class CLIPFeatureExtractor:
    """
    Frozen CLIP feature extractor for scene-graph nodes and edges.

    CLIP is used purely as a fixed embedding function; no weights are updated.
    L2-normalising embeddings keeps dot products interpretable as cosine similarity.

    Usage:
        fe = CLIPFeatureExtractor(device="cuda")
        node_feats = fe.extract_node_features(image, boxes_xyxy, labels)
        edge_feats, edge_index = fe.extract_edge_features(
            image, boxes_xyxy, node_feats, triplets)
    """

    def __init__(self, model_id=CLIP_MODEL_ID, device="cuda"):
        self.device    = device
        self.processor = CLIPProcessor.from_pretrained(model_id, use_fast=True)
        self.model     = CLIPModel.from_pretrained(model_id,
        ).to(device)

        
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def extract_node_features(self, image: Image.Image,
                               boxes_xyxy: torch.Tensor,
                               labels: List[str]) -> torch.Tensor:
        """
        Compute one NODE_DIM feature vector per detected object.

        Steps: crop → CLIP visual embed, class label → CLIP text embed,
               normalised box → 8-d geometry. All concatenated to [N, 1032].

        DEBUG: verify NODE_DIM matches the constant imported in gnn.py.
        """
        
        n    = len(labels)
        if n == 0:
            return torch.zeros((0, NODE_DIM), device=self.device)
        W, H = image.size

        
        boxes_norm = torch.stack([_norm_box(boxes_xyxy[i], W, H) for i in range(n)])

        
        crops = [_crop(image, boxes_xyxy[i]) for i in range(n)]
        vis_inputs = self.processor(images=crops, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            
            f_v_raw = self.model.get_image_features(**vis_inputs)
            
            
            f_v = F.normalize(f_v_raw.pooler_output, p=2, dim=-1)
        print(f_v.shape)
        
        text_inputs = self.processor(
            text=labels, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)

        f_t_output = self.model.get_text_features(**text_inputs)

        f_t = F.normalize(f_t_output.pooler_output, dim=-1)
        print(f_t.shape)
        bbox = _bbox_8d(boxes_norm).to(self.device)  
        return torch.cat([f_v, f_t, bbox], dim=-1)   

    @torch.no_grad()
    def extract_edge_features(self, image: Image.Image,
                               boxes_xyxy: torch.Tensor,
                               node_feats: torch.Tensor,
                               triplets: List[Tuple[int, str, int]],
                               node_labels: List[str] = None, 
                               projector: Optional[nn.Module] = None):
        """
        Compute one EDGE_DIM feature vector per proposed (subject, predicate, object) edge.

        Batching: union crops and predicate strings are both processed in a single
        CLIP forward pass each for efficiency.

        Returns:
            edge_feats  Tensor[E, EDGE_DIM]
            edge_index  Tensor[2, E]

        DEBUG: if EDGE_DIM mismatches gnn.py, verify the concat order above.
        """
        
        if not triplets:
            
            return (torch.zeros(0, EDGE_DIM, device=self.device),
                    torch.zeros(2, 0, dtype=torch.long, device=self.device))

        W, H       = image.size
        boxes_norm = torch.stack([_norm_box(boxes_xyxy[i], W, H)
                                  for i in range(len(node_feats))])

        
        union_crops = [_crop(image, _union_box(boxes_xyxy[s], boxes_xyxy[o]))
                       for s, _, o in triplets]
        vis_outputs = self.model.get_image_features(
            **self.processor.image_processor(
                images=union_crops, return_tensors="pt"
            ).to(self.device)
        )
        
        
        union_vis = F.normalize(vis_outputs.pooler_output, dim=-1)

        
        
        
        
        
        if node_labels is not None:
            rel_texts = [f"a photo of a {node_labels[s]} {r} a {node_labels[o]}" for s, r, o in triplets]
        else:
            rel_texts = [r for _, r, _ in triplets]

        outputs = self.model.get_text_features(
            **self.processor.tokenizer(
                text=rel_texts, return_tensors="pt", 
                padding=True, truncation=True, max_length=77
            ).to(self.device)
        )
        

        
        if projector is not None:
            
            with torch.enable_grad():
                rel_text = F.normalize(projector(outputs.pooler_output), dim=-1)
        else:
            rel_text = F.normalize(outputs.pooler_output, dim=-1)

        edges, src_list, dst_list = [], [], []
        for idx, (s, _, o) in enumerate(triplets):
            geom = _geom_8d(boxes_norm[s], boxes_norm[o]).to(self.device)  
            edges.append(torch.cat([
                union_vis[idx],   
                geom,             
                node_feats[s],    
                node_feats[o],    
                rel_text[idx],    
            ]))                   
            src_list.append(s)
            dst_list.append(o)

        return (
            torch.stack(edges),                                     
            torch.tensor([src_list, dst_list], dtype=torch.long),  
        )