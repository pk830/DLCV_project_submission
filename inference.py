"""
inference.py — Standalone inference script for the Ov-SGG system.

Loads the trained GNN checkpoint (gnn_refiner.pth) and runs the full
inference pipeline on one or more images:

    Detect (Grounding DINO) → NMS → Propose (Qwen3-VL)
        → CLIP features → GNN filter → final scene-graph triplets

Usage:
    python inference.py --config config.yaml \\
                        --images img1.jpg img2.jpg \\
                        --labels "man . bike . tree . road" \\
                        --visualise          

Output:
    For each image, prints the accepted (subject, predicate, object) triplets
    and optionally saves a debug visualisation to <image_stem>_sgg.jpg.
"""

from __future__ import annotations
import argparse
import yaml
from pathlib import Path

import torch
from PIL import Image

from vlm import GroundingDINODetector, QwenVLProposer
from features import CLIPFeatureExtractor
from gnn import GNNRefiner
from pipeline import OvSGGPipeline


def load_pipeline(cfg: dict, device: str) -> OvSGGPipeline:
    """
    Instantiate and return a fully initialised OvSGGPipeline with the
    trained GNN weights loaded from cfg["save_dir"]/gnn_refiner.pth.

    All models are placed on `device`. The GNN is set to eval mode.
    """
    detector = GroundingDINODetector(
        cfg.get("gdino_model_id", "IDEA-Research/grounding-dino-base"),
        device=device)

    proposer = QwenVLProposer(
        cfg.get("qwen_model_id", "Qwen/Qwen3-VL-8B-Instruct"),
        device=device)

    feature_extractor = CLIPFeatureExtractor(device=device)

    gnn = GNNRefiner(
        hidden_dim=cfg.get("hidden_dim", 256),
        num_sage_layers=cfg.get("num_sage_layers", 2),
        dropout=cfg.get("dropout", 0.1),
    ).to(device)

    
    ckpt_path = Path(cfg.get("save_dir", "./models")) / "gnn_refiner.pth"
    ckpt      = torch.load(ckpt_path, map_location=device)
    gnn.load_state_dict(ckpt["gnn_state"])
    gnn.eval()
    print(f"[inference] Loaded GNN checkpoint from {ckpt_path} "
          f"(epoch={ckpt['epoch']}, best_f1={ckpt['best_f1']:.4f})")

    return OvSGGPipeline(
        detector=detector,
        proposer=proposer,
        feature_extractor=feature_extractor,
        gnn=gnn,
        nms_iou_thr=cfg.get("nms_iou_thr", 0.5),
        edge_score_thr=cfg.get("edge_score_thr", 0.5),
        device=device,
    )


def run_inference(pipeline: OvSGGPipeline,
                  image_paths: list,
                  label_prompts: str,
                  visualise: bool = False):
    """
    Run forward_inference on each image and print the resulting scene graph.

    Args:
        pipeline      : Initialised OvSGGPipeline.
        image_paths   : List of image file path strings.
        label_prompts : Dot-separated category string for Grounding DINO,
                        e.g. "man . bike . tree . road".
                        Use broad categories to maximise recall.
        visualise     : If True, save an annotated image alongside each input.
    """
    for img_path in image_paths:
        image  = Image.open(img_path).convert("RGB")
        output = pipeline.forward_inference(image, label_prompts)

        print(f"\n{'='*60}")
        print(f"Image : {img_path}")
        print(f"Objects detected ({len(output['labels'])}):")
        for i, lbl in enumerate(output["labels"]):
            score = output["scores"][i].item()
            print(f"  [{i}] {lbl}  (conf={score:.2f})")

        print(f"\nAccepted triplets ({len(output['final_triplets'])}) "
              f"[{len(output['proposals'])} proposed]:")
        for s, rel, o in output["final_triplets"]:
            s_lbl = output["labels"][s]
            o_lbl = output["labels"][o]
            
            edge_idx = output["proposals"].index((s, rel, o))
            score    = output["edge_scores"][edge_idx].item()
            print(f"  ({s_lbl}, {rel}, {o_lbl})  [score={score:.3f}]")

        if visualise and output["final_triplets"]:
            
            annotated  = pipeline.visualise(image, output)
            save_path  = Path(img_path).with_stem(Path(img_path).stem + "_sgg")
            annotated.save(save_path)
            print(f"  Saved visualisation → {save_path}")


def main():
    p = argparse.ArgumentParser(description="Ov-SGG inference")
    p.add_argument("--config",    default="config.yaml",
                   help="Path to config.yaml used during training.")
    p.add_argument("--images",    nargs="+", required=True,
                   help="One or more image file paths to run inference on.")
    p.add_argument("--labels",    default="person . car . chair . table . dog . cat",
                   help="Dot-separated category prompt for Grounding DINO.")
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--visualise", action="store_true",
                   help="Save annotated output images alongside inputs.")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pipeline = load_pipeline(cfg, args.device)
    run_inference(pipeline, args.images, args.labels, args.visualise)


if __name__ == "__main__":
    main()