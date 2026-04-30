"""
pipeline.py — OvSGGPipeline: central integration class for Ov-SGG.

Training mode  (forward_train):
    Image → Grounding DINO (detect_train, with grad) → NMS
        → match detected boxes to GT boxes (for box regression loss)
        → VLM propose (on detected boxes + labels)
        → label VLM proposals against GT triplets (-1.0 / 0.0 / 1.0)
        → CLIP features → build PyG graph → return for GNN loss

    Grounding DINO is called with a text prompt built from the GT label set.
    Its raw outputs are returned so train.py can compute the box regression loss.

Inference mode (forward_inference):
    Image → Grounding DINO detect() (no_grad) → NMS
        → VLM propose → CLIP features → GNN filter → final triplets.
"""

from __future__ import annotations
import random
import torch
import torchvision.ops as tv_ops
from typing import List, Tuple, Dict, Any, Optional, Callable
from PIL import Image, ImageDraw
import torch.nn.functional as F
import torch.nn as nn
from vlm import GroundingDINODetector, QwenVLProposer
from features import CLIPFeatureExtractor
from gnn import GraphBuilder, GNNRefiner, NODE_DIM, EDGE_DIM




from torch_geometric.data import Data
import pdb







def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert [N, 4] (x, y, w, h) → (x1, y1, x2, y2).
    """
    return torch.stack([
        boxes[:, 0], boxes[:, 1],
        boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3],
    ], dim=-1)


def apply_nms(boxes: torch.Tensor, scores: torch.Tensor,
              iou_thr: float = 0.5) -> torch.Tensor:
    """Class-agnostic NMS. Returns kept indices."""
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long)
    return tv_ops.nms(boxes.float(), scores.float(), iou_thr)


def normalise_boxes(boxes: torch.Tensor, W: int, H: int) -> torch.Tensor:
    """Normalise xyxy pixel boxes to [0,1] range given image width W and height H."""
    scale = torch.tensor([W, H, W, H], dtype=boxes.dtype, device=boxes.device)
    return boxes / scale





def _build_dino_prompt(gt_labels: List[str],
                       object_vocab: Optional[List[str]]) -> str:
    """
    Build the dot-separated DINO text prompt.
    """
    unique_gt = list(dict.fromkeys(gt_labels))
    if object_vocab:
        gt_set      = set(unique_gt)
        candidates  = [o for o in object_vocab if o not in gt_set]
        k           = min(len(unique_gt), len(candidates))
        distractors = random.sample(candidates, k) if k > 0 else []
        combined    = unique_gt + distractors
        random.shuffle(combined)
    else:
        combined = unique_gt
    return " . ".join(combined)






class OvSGGPipeline:
    """
    Central integration class for the Ov-SGG system.
    """

    def __init__(self, detector: GroundingDINODetector,
                 proposer: QwenVLProposer,
                 feature_extractor: CLIPFeatureExtractor,
                 gnn: GNNRefiner,
                 projector: Optional[nn.Module] = None,
                 graph_builder: Optional[GraphBuilder] = None,
                 nms_iou_thr: float = 0.5,
                 edge_score_thr: float = 0.5,
                 predicate_vocab: Optional[List[str]] = None,
                 object_vocab: Optional[List[str]] = None,
                 spatial_overlap_thr: float = 0.0,
                 spatial_dist_thr: float = 0.85,
                 neg_keep_ratio: float = 3.0,
                 no_relation_predicate: str = "no relation",
                 gt_match_iou_floor: float = 0.0,
                 enable_relation_pruning: bool = True,
                 device: str = "cuda"):
        self.detector                = detector
        self.proposer                = proposer
        self.feature_extractor       = feature_extractor
        self.gnn                     = gnn
        self.projector                = projector 
        self.graph_builder           = graph_builder or GraphBuilder()
        self.nms_iou_thr             = nms_iou_thr
        self.edge_score_thr          = edge_score_thr
        self.predicate_vocab         = predicate_vocab  
        self.object_vocab            = object_vocab      
        self.spatial_overlap_thr     = spatial_overlap_thr
        self.spatial_dist_thr        = spatial_dist_thr
        self.neg_keep_ratio          = neg_keep_ratio
        self.no_relation_predicate   = no_relation_predicate
        self.gt_match_iou_floor      = gt_match_iou_floor
        self.enable_relation_pruning = enable_relation_pruning
        self.device                  = device
        

    
    
    

    def forward_train(self, image: Image.Image,
                      gt_boxes_xyxy: torch.Tensor,
                      gt_labels: List[str],
                      gt_triplets: List[Tuple[int, str, int]],
                      teacher_forcing: bool = False,
                      cached_triplets: Optional[List[Tuple[int, str, int]]] = None) -> Dict[str, Any]: 
        """
        Training forward pass utilizing the offline VLM Cache.
        """
        W, H = image.size
        gt_boxes_xyxy = gt_boxes_xyxy.to(self.device)

        
        text_prompt = _build_dino_prompt(gt_labels, self.object_vocab)
        det = self.detector.detect_train(image, text_prompt)

        pred_boxes_raw     = det["outputs"].pred_boxes[0]          
        gt_boxes_xyxy_norm = normalise_boxes(gt_boxes_xyxy, W, H)  

        assert pred_boxes_raw.requires_grad, \
            "[ASSERT] pred_boxes_raw.requires_grad is False — DINO decoder gradient is disconnected."

        
        if teacher_forcing:
            gnn_boxes  = gt_boxes_xyxy
            gnn_labels = gt_labels
        else:
            keep      = apply_nms(det["boxes"], det["scores"], self.nms_iou_thr)
            det_boxes = det["boxes"][keep]
            if det_boxes.shape[0] == 0:
                return self._empty_train(pred_boxes_raw, gt_boxes_xyxy_norm, det["outputs"])
            gnn_boxes  = det_boxes.to(self.device).detach()
            gnn_labels = [det["labels"][i] for i in keep.tolist()]
        
        N = gnn_boxes.shape[0]
        
        if cached_triplets is not None:
            
            iou_matrix = tv_ops.box_iou(gnn_boxes, gt_boxes_xyxy) 
            max_iou, gt_match_indices = iou_matrix.max(dim=1)
            
            
            cache_dict = {(t[0], t[2]): t[1] for t in cached_triplets}
            proposals = []
            
            for i in range(N):
                for j in range(N):
                    if i == j: continue
                    
                    predicate = self.no_relation_predicate
                    
                    
                    idx_i = gt_match_indices[i].item()
                    idx_j = gt_match_indices[j].item()
                    
                    
                    if max_iou[i] > 0.5 and max_iou[j] > 0.5:
                        if (idx_i, idx_j) in cache_dict:
                            predicate = cache_dict[(idx_i, idx_j)]
                    
                    proposals.append((i, predicate, j))
                    
            filter_stats = {"vlm_bypassed_using_cache": True}
        else:
            
            proposals = self.proposer.propose_relations(
                image, gnn_labels, gnn_boxes, predicate_vocab=self.predicate_vocab
            )
            filter_stats = {}

        if not proposals:
            return self._empty_train(pred_boxes_raw, gt_boxes_xyxy_norm, det["outputs"])
        
        
        node_feats = self.feature_extractor.extract_node_features(
            image, gnn_boxes, gnn_labels)
            
        edge_feats, edge_index = self.feature_extractor.extract_edge_features(
            image, gnn_boxes, node_feats, proposals, 
            node_labels=gnn_labels, 
            projector=self.projector 
        )

        
        graph = self.graph_builder.build(node_feats, edge_feats, edge_index)

        
        return {
            "graph":                 graph,
            "pred_boxes_raw":        pred_boxes_raw,      
            "gt_boxes_xyxy_norm":    gt_boxes_xyxy_norm,  
            "det_labels":            gnn_labels,
            "proposals":             proposals,
            "proposal_filter_stats": filter_stats,
            "dino_outputs":          det["outputs"],
        } 
    


    @torch.no_grad()
    def forward_inference(self, image: Image.Image,
                          label_prompts: str,
                          gt_labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Full inference pipeline designed for maximum Recall@K (Multi-graph enabled).
        """
        text_prompt = (_build_dino_prompt(gt_labels, self.object_vocab)
                       if gt_labels is not None else label_prompts)
        det    = self.detector.detect(image, text_prompt)
        boxes  = det["boxes"]
        scores = det["scores"]
        labels = det["labels"]

        keep   = apply_nms(boxes, scores, self.nms_iou_thr)
        boxes  = boxes[keep]
        scores = scores[keep]
        labels = [labels[i] for i in keep.tolist()]

        if not labels:
            return self._empty_inference()

        proposals = self.proposer.propose_relations(
            image,
            labels,
            boxes,
            predicate_vocab=self.predicate_vocab,
        )

        if self.enable_relation_pruning:
            proposals, _, _ = filter_proposals_with_spatial_and_neg_sampling(
                proposals=proposals,
                boxes_xyxy=boxes,
                image_size=image.size,
                spatial_overlap_thr=self.spatial_overlap_thr,
                spatial_dist_thr=self.spatial_dist_thr,
                neg_keep_ratio=0.0,
                no_relation_predicate=self.no_relation_predicate,
            )

        if not proposals:
            return self._empty_inference(boxes, labels, scores)

        node_feats = self.feature_extractor.extract_node_features(image, boxes, labels)
        edge_feats, edge_index = self.feature_extractor.extract_edge_features(
            image, boxes, node_feats, proposals,
            node_labels=labels,           
            projector=self.projector      
        )

        graph = self.graph_builder.build(node_feats, edge_feats, edge_index)
        self.gnn.eval()
        edge_scores    = torch.sigmoid(self.gnn(graph.to(self.device))).cpu()
        
        
        final_triplets = [t for t, s in zip(proposals, edge_scores.tolist())
                          if s >= self.edge_score_thr and t[1] != self.no_relation_predicate]

        return {"boxes_xyxy": boxes, "labels": labels, "scores": scores,
                "proposals": proposals, "edge_scores": edge_scores,
                "final_triplets": final_triplets, "graph": graph}

    
    
    

    def _empty_train(self, pred_boxes_raw: torch.Tensor = None,
                     gt_boxes_xyxy_norm: torch.Tensor = None,
                     dino_outputs=None) -> Dict[str, Any]:
        g = Data(x=torch.zeros(0, NODE_DIM),
                 edge_index=torch.zeros(2, 0, dtype=torch.long),
                 edge_attr=torch.zeros(0, EDGE_DIM)) 
        return {
            "graph":               g,
            "pred_boxes_raw":      pred_boxes_raw     if pred_boxes_raw is not None else torch.zeros(0, 4),
            "gt_boxes_xyxy_norm":  gt_boxes_xyxy_norm if gt_boxes_xyxy_norm is not None else torch.zeros(0, 4),
            "det_labels":          [],
            "proposals":           [],
            "dino_outputs":        dino_outputs,
        }

    def _empty_inference(self, boxes=None, labels=None, scores=None) -> Dict[str, Any]:
        return {
            "boxes_xyxy":     boxes  if boxes  is not None else torch.zeros(0, 4),
            "labels":         labels if labels is not None else [],
            "scores":         scores if scores is not None else torch.zeros(0),
            "proposals":      [],
            "edge_scores":    torch.zeros(0),
            "final_triplets": [],
            "graph":          None,
        }

    
    
    

    def visualise(self, image: Image.Image,
                  output: Dict[str, Any]) -> Image.Image:
        img  = image.copy()
        draw = ImageDraw.Draw(img)

        for i, (box, lbl) in enumerate(zip(output["boxes_xyxy"], output["labels"])):
            x1, y1, x2, y2 = box.int().tolist()
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1 - 12), f"[{i}] {lbl}", fill="red")

        
        triplets_to_draw = output.get("final_triplets", output.get("proposals", []))

        for s, rel, o in triplets_to_draw:
            if rel == self.no_relation_predicate:
                continue 
                
            bs, bo = output["boxes_xyxy"][s], output["boxes_xyxy"][o]
            cs  = ((bs[0] + bs[2]) / 2, (bs[1] + bs[3]) / 2)
            co  = ((bo[0] + bo[2]) / 2, (bo[1] + bo[3]) / 2)
            draw.line([cs[0].item(), cs[1].item(),
                       co[0].item(), co[1].item()], fill="blue", width=2)
            mid = ((cs[0] + co[0]) / 2, (cs[1] + co[1]) / 2)
            draw.text((mid[0].item(), mid[1].item()), rel, fill="blue")

        return img