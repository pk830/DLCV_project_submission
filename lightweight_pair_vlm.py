import pdb
pdb.set_trace()
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import re
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
import itertools
import numpy as np
from itertools import combinations
from collections import defaultdict
# from groundingdino.util.inference import annotate  # optional for visualization

class EntityMapper:
    def __init__(self, 
                 gt_vocab: List[str], 
                 simcse_model: str = "princeton-nlp/sup-simcse-roberta-large", 
                 device: str = "cuda"):
        
        self.device = device
        self.gt_vocab = [label.strip().lower() for label in gt_vocab]
        
        print(f"Loading SimCSE model: {simcse_model}")
        self.encoder = SentenceTransformer(simcse_model, device=device)
        
        # Precompute GT embeddings
        self.gt_sentences = [f"There is a {label} in the image." for label in self.gt_vocab]
        self.gt_embeddings = self.encoder.encode(
            self.gt_sentences, 
            convert_to_tensor=True, 
            normalize_embeddings=True
        )

    def normalize_label(self, label: str) -> str:
        if not label:
            return ""
        normalized = label.strip().lower()
        normalized = re.sub(r'[^\w\s]', '', normalized)  # remove punctuation
        return normalized

    def map_entities(self, 
                     vlm_labels: List[str], 
                     top_k: int = 2, 
                     delta: float = 0.05, 
                     tau: float = 0.2,
                     min_cosine_threshold: float = 0.7):   # ← New parameter
        
        results = []
        
        for raw_label in vlm_labels:
            norm_label = self.normalize_label(raw_label)
            if not norm_label:
                continue

            mapping = {
                "raw_vlm": raw_label,
                "normalized": norm_label,
                "exact_match": None,
                "semantic_matches": []
            }

            # Stage 2: Exact match (still highest priority)
            if norm_label in self.gt_vocab:
                idx = self.gt_vocab.index(norm_label)
                mapping["exact_match"] = {
                    "gt_label": self.gt_vocab[idx],
                    "confidence": 1.0
                }
                results.append(mapping)
                continue

            # Stage 3: Semantic matching with new minimum cosine threshold
            candidate_sentence = f"There is a {norm_label} in the image."
            cand_emb = self.encoder.encode(
                candidate_sentence, 
                convert_to_tensor=True, 
                normalize_embeddings=True
            ).unsqueeze(0)

            # Cosine similarity (raw, before temperature)
            sim_scores = F.cosine_similarity(cand_emb, self.gt_embeddings, dim=1)  # (num_gt,)

            # === NEW FILTER: Keep only scores >= 0.7 ===
            valid_mask = sim_scores >= min_cosine_threshold
            if not valid_mask.any():
                # No semantic match meets the 0.7 threshold → discard this prediction
                mapping["semantic_matches"] = []
                results.append(mapping)
                continue

            # Apply filter
            filtered_sim = sim_scores[valid_mask]
            filtered_indices = torch.nonzero(valid_mask).squeeze(-1)

            # Temperature scaling on the filtered scores only
            scaled_scores = filtered_sim / tau
            max_scaled = scaled_scores.max().item()

            # Delta filtering relative to the new max
            delta_mask = scaled_scores >= (max_scaled - delta)
            final_scores = scaled_scores[delta_mask]
            final_indices = filtered_indices[delta_mask]

            # Take top-k
            if len(final_scores) > 0:
                topk_values, topk_idx = torch.topk(final_scores, min(top_k, len(final_scores)))
                topk_gt_indices = final_indices[topk_idx]

                for val, gidx in zip(topk_values, topk_gt_indices):
                    mapping["semantic_matches"].append({
                        "gt_label": self.gt_vocab[gidx.item()],
                        "cosine_score": sim_scores[gidx].item(),   # original raw cosine
                        "scaled_score": val.item()
                    })

            results.append(mapping)
        
        return results
    


class GroundingDINODetector:
    def __init__(self, 
                 model_id="IDEA-Research/grounding-dino-tiny", 
                 device="cuda"):
        self.device = device
        # Load the HF Processor and Model
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(device)
        
        self.box_threshold = 0.35
        self.text_threshold = 0.25

    def detect_single_object(self, image_path: str, object_name: str) -> List[Dict]:
        """Detect object using Hugging Face implementation"""
        # Load image via PIL (required for HF processor)
        image = Image.open(image_path).convert("RGB")
        
        # Grounding DINO expects the text prompt to end with a dot
        text_prompt = object_name.lower().strip() + "."

        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process detections
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]] # (height, width)
        )[0]

        detections = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            detections.append({
                "bbox": box.tolist(),  # [xmin, ymin, xmax, ymax]
                "score": score.item(),
                "label": label
            })
        return detections
    
def get_semantic_pair_refinement_prompt(candidate_pairs: List[Tuple[str, str]]) -> str:
    """
    Returns the prompt in the exact style you requested.
    """
    pair_list = []
    for i, (subj, obj) in enumerate(candidate_pairs, 1):
        pair_list.append(f"Pair {i}: {subj} and {obj}")
    
    pair_text = "\n".join(pair_list)

    prompt = f"""You are a world-class vision-language analyst, highly specialized in understanding spatial and functional relationships between objects in visual scenes. Your role is to evaluate how likely it is that specific object pairs are engaged in meaningful physical interactions in the given image.

### Object Pair List:
{pair_text}

### Task:
Carefully assess each object pair listed above and determine the likelihood that they participate in a meaningful interaction within the scene. Base your assessment on how objects of those categories typically relate in physical or functional terms within real-world images.

Provide a single integer confidence score from 1 to 5 for each pair, where:
- 1 = Very Unlikely
- 2 = Unlikely
- 3 = Uncertain
- 4 = Likely
- 5 = Very Likely

### Output Format:
- Do not include any object names, explanations, or extra text.
- Stop after the final pair.
- You must return exactly one line per pair listed above.
- Use the format: Pair [index]: [score]

### Begin:"""

    return prompt

# default: Load the model on the available device(s)
#pdb.set_trace()
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", dtype=torch.bfloat16, device_map="cuda"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-8B-Instruct",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", use_fast=True)
# prompt_text = """
# ### Task Start
# You are an expert at detecting objects in images. You are given an image.Your task is to list all identifiable objects visible in the image, including those in the foreground and background. Include both whole objects and meaningful parts or components that are visually discernible.
# ### Output Format Instructions
# - Do not repeat object names.
# - Do not describe attributes, adjectives,
# or relationships.
# - Return the result as a comma-separated
# list.
# - If unsure, do not include it.
# ### Prompt
# List all the objects visible in the image, including foreground and background. Return the objects as a comma-separated list."""

prompt_text = """You are an accurate object detector. 
Look at the image carefully and list ONLY the objects that are clearly visible.

Rules:
- ONLY include objects you can actually see.
- Do NOT guess, infer, or add objects that might be there but are not visible.
- Do NOT include materials (wood, metal, glass, fabric, etc.), parts of objects unless they are standalone, or abstract concepts.
- Do NOT include generic things like "street", "road", "shadow", "light", "building" unless they are clearly the main subject.
- Return ONLY a simple comma-separated list with no duplicates, no explanations, and no extra text.

Example output: person, car, traffic light, bench, tree, sidewalk, pole
"""

image_path = "/home/venky/koushikpavan/gqa/gqa/images/2364339.jpg"
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {"type": "text", "text": prompt_text},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(
    **inputs,
    max_new_tokens=512,
    #do_sample=True,
    temperature=0.1,           # higher than 0.1
    top_p=0.95,                # never use 1.0 with low temperature
    repetition_penalty=1.1,   # stronger penalty
)


input_len = inputs["input_ids"].shape[1]
generated_ids_trimmed = generated_ids[:, input_len:]

output_text = processor.batch_decode(
    generated_ids_trimmed, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=True
)[0]

print(output_text)
#pdb.set_trace()
objects = [obj.strip() for obj in output_text.split(',') if obj.strip()]
unique_objects = []
seen = set()
for obj in objects:
    if obj.lower() not in seen:
        seen.add(obj.lower())
        unique_objects.append(obj)

#pdb.set_trace()
gt_vocab = [
    "mountain", "cow", "people", "face", "number", "pizza", "tire", "player", "pillow",
    "screen", "truck", "kite", "trunk", "sock", "neck", "glove", "coat", "letter", "roof",
    "windshield", "desk", "paw", "leaf", "flower", "plant", "counter", "paper", "eye",
    "book", "branch", "lamp", "cup", "phone", "toilet", "skateboard", "logo", "laptop",
    "vehicle", "motorcycle", "hill", "curtain", "nose", "sheep", "bowl", "wire", "bear",
    "banana", "mouth", "drawer", "shelf", "cap", "animal", "bottle", "box", "airplane",
    "finger", "room", "flag", "seat", "tower", "wing", "fruit", "rock", "house", "pot", "bird",
    "umbrella", "surfboard", "lady", "tie", "fork", "vase", "bag", "orange", "clock",
    "sidewalk", "food", "sink", "cabinet", "beach", "boat", "basket", "helmet", "child",
    "racket", "post", "guy", "towel", "arm", "napkin", "bush", "bench", "person", "cone",
    "apple", "jacket", "fur", "air", "sign", "bus", "wrist", "frame", "floor", "dress", "street",
    "shoe", "ball", "girl", "ear", "boy", "broccoli", "fence", "uniform", "hair", "sneakers",
    "blanket", "zebra", "train", "camera", "sticker", "license plate", "lid", "tomato",
    "pants", "giraffe", "watch", "wall", "leg", "bed", "t-shirt", "shorts", "horse", "spots",
    "arrow", "field", "bread", "bicycle", "knife", "couch", "ceiling",

    "ocean", "car", "picture", "hand", "snow", "horn", "woman", "sweater", "container",
    "paint", "feet", "clouds", "foot", "dirt", "faucet", "chair", "sand", "tail", "stone", "cat",
    "tag", "traffic light", "keyboard", "tree", "leaves", "elephant", "ground", "glass",
    "frisbee", "trash can", "word", "man", "jeans", "door", "building", "sky", "table",
    "wheel", "pole", "collar", "hat", "cheese", "mane", "shirt", "dog", "cord", "cake",
    "donut", "plate", "backpack", "mirror", "street light", "skis", "window", "grass",
    "water", "bike", "road", "head", "cell phone"
]

mapper = EntityMapper(gt_vocab=gt_vocab, device="cuda")

# Example VLM predictions (from Qwen3-VL)
vlm_predictions = unique_objects

mappings = mapper.map_entities(
    vlm_predictions, 
    top_k=2, 
    delta=0.05, 
    tau=0.2
)
detector = GroundingDINODetector()
final_detections = {}
# Print results

for m in mappings:
    print(f"VLM: {m['raw_vlm']} → Normalized: {m['normalized']}")
    if m["exact_match"]:
        gt_label = m["exact_match"]["gt_label"]
    elif m["semantic_matches"]:
        for match in m["semantic_matches"]:
            print(f"  Semantic: {match['gt_label']} (cosine={match['cosine_score']:.4f})")
        gt_label = m["semantic_matches"][0]["gt_label"]
    else:
        print("  → Discarded (no match above 0.7 cosine)")
        continue
    print("-" * 60)
    boxes = detector.detect_single_object(image_path, gt_label)
    
    final_detections[gt_label] = boxes



def get_center(box):
    """box = [x1, y1, x2, y2]"""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return np.array([cx, cy])

def compute_iou(box1, box2):
    """Compute Intersection over Union between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    inter_area = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

with Image.open(image_path) as img:
    W, H = img.size
diag = np.sqrt(H**2 + W**2)

# 2. Collect all instances (Fixing the dictionary access)
instances = []
for label, boxes in final_detections.items():
    for idx, box_info in enumerate(boxes):
        # Handle both raw list or dict from detector
        actual_box = box_info['bbox'] if isinstance(box_info, dict) else box_info
        
        instances.append({
            "label": label,
            "instance_id": idx,
            "box": actual_box,
            "center": get_center(actual_box)
        })

candidate_pairs = []
all_info_candidate_pairs = []
for inst_i, inst_j in itertools.combinations(instances, 2):
    pair = {
        "sub_label": inst_i["label"],
        "sub_id": inst_i["instance_id"],
        "obj_label": inst_j["label"],
        "obj_id": inst_j["instance_id"],
        "sub_box": inst_i["box"],
        "obj_box": inst_j["box"]
    }
    candidate_pairs.append((inst_i["label"],inst_j["label"]))
    all_info_candidate_pairs.append(pair)

lambda1 = 1.0     # Weight for 2D distance
tau = 0.5         # Distance threshold (0.0 to 1.0)
beta = 10.0       # Inverse temperature (controls sharpness of the "drop-off")

def sigmoid(x):
    # Standard sigmoid function
    return 1 / (1 + np.exp(-x))

compatible_pairs = []
pair_score_dict = {}

# 1. Iterate through pairs of grounded objects
for inst_i, inst_j in combinations(instances, 2):
    c_i, c_j = inst_i["center"], inst_j["center"]

    # 2. Compute Euclidean distance between 2D centers (x_ij)
    x_ij = np.linalg.norm(c_i - c_j)
    
    # 3. Normalize by image diagonal (y)
    # This represents the (x_ij / y) term from Eq. 2
    norm_2d_dist = x_ij / diag

    # 4. Total Distance d_ij (Eq. 2 without depth)
    # Since depth is out, d_ij is just the weighted 2D distance
    d_ij = lambda1 * norm_2d_dist

    # 5. Thresholding (Retain pair if d_ij < tau)
    if d_ij < tau:
        # 6. Compatibility Score P^G_ij (Eq. 3)
        # Higher score (closer to 1.0) means objects are closer together
        geom_score = sigmoid(-beta * (d_ij - tau))

        pair_idx = len(compatible_pairs) + 1
        pair_key = f"pair_{pair_idx}"
        
        pair_data = {
            "pair_id": pair_key,
            "sub_label": inst_i["label"],
            "obj_label": inst_j["label"],
            "norm_dist": round(float(norm_2d_dist), 4),
            "geom_score": round(float(geom_score), 4) # This is P^G_ij
        }
        
        compatible_pairs.append(pair_data)
        pair_score_dict[pair_key] = pair_data["geom_score"]

# Example Output
print(f"Filtered to {len(compatible_pairs)} spatially plausible pairs.")


prompt_text = get_semantic_pair_refinement_prompt(candidate_pairs)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt_text},
        ]
    }
]

# Prepare inputs
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, 
    return_dict=True, return_tensors="pt"
).to(model.device)

# Generate
generated_ids = model.generate(
    **inputs,
    max_new_tokens=1024,
    temperature=0.1,           # higher than 0.1
    top_p=0.95,                # never use 1.0 with low temperature
    repetition_penalty=1.1,
)

# Decode
input_len = inputs["input_ids"].shape[1]
output_text = processor.batch_decode(
    generated_ids[:, input_len:], 
    skip_special_tokens=True
)[0]

matches = re.findall(r'Pair (\d+): (\d+)', output_text)

# Create the dictionary
# Converting to int for cleaner data, or keep as string if preferred
semantic_score_dict = {f"pair_{pair_num}": int(score) for pair_num, score in matches}
bbox_score_dict = pair_score_dict
#all_info_candidate_pairs = all_info_candidate_pairs

def fuse_semantic_and_geometric(
    semantic_dict: dict,
    geometric_dict: dict,
    alpha: float = 0.6,
    top_k: int = 150,
    eps: float = 1e-6
):
    """
    Fuse semantic (1-5) and geometric (0-1) scores using the paper's log-weighted formula.
    """
    combined_scores = {}
    common_keys = set(semantic_dict.keys()) & set(geometric_dict.keys())
    
    print(f"Found {len(common_keys)} common pairs for fusion.")

    for key in common_keys:
        sem_raw = semantic_dict[key]      # 1 to 5
        geo_raw = geometric_dict[key]     # 0 to 1
        
        # === Normalize both to [0,1] range first (very important!) ===
        ps = (sem_raw - 1) / 4.0          # maps 1→0, 5→1.0
        pg = float(geo_raw)
        
        # Avoid log(0)
        ps = max(ps, eps)
        pg = max(pg, eps)
        
        # Paper's formula: weighted sum of log probabilities
        p_combined = alpha * np.log(ps) + (1 - alpha) * np.log(pg)
        
        combined_scores[key] = {
            "combined_score": p_combined,
            "semantic_raw": sem_raw,
            "geometric_raw": pg,
            "semantic_norm": round(ps, 4),
            "geometric_norm": round(pg, 4),
            "key": key
        }
    
    # Sort descending by combined score
    sorted_pairs = sorted(
        combined_scores.values(),
        key=lambda x: x["combined_score"],
        reverse=True
    )
    
    top_pairs = sorted_pairs[:top_k]
    
    print(f"\nFusion completed (alpha={alpha}):")
    print(f"→ Selected top {len(top_pairs)} candidate pairs\n")
    
    # Show top 15 results
    print("Top 15 fused pairs:")
    for i, item in enumerate(top_pairs[:15]):
        print(f"{i+1:2d}. {item['key']:35}  "
              f"Combined={item['combined_score']:.4f}   "
              f"Sem={item['semantic_raw']}({item['semantic_norm']})   "
              f"Geo={item['geometric_raw']:.3f}")
    
    return top_pairs


# ====================== USAGE ======================
pdb.set_trace()
alpha = 0.65     # Recommended starting point (0.5 ~ 0.7)
top_k = 40      # Adjust based on how many candidate pairs you want

final_top_pairs = fuse_semantic_and_geometric(
    semantic_dict=semantic_score_dict,
    geometric_dict=bbox_score_dict,   # your pair_score_dict
    alpha=alpha,
    top_k=top_k
)

def get_vlm_triplet_prompt(candidate_pairs_with_boxes: List[Dict]) -> str:
    """
    Constructs a prompt for a VLM to generate (Subject, Predicate, Object) triplets
    using normalized bounding box coordinates.
    """
    pair_entries = []
    for i, p in enumerate(candidate_pairs_with_boxes, 1):
        # Formatting the boxes into the [x1, y1, x2, y2] style
        sub_box = [int(x) for x in p['sub_box']]
        obj_box = [int(x) for x in p['obj_box']]
        
        entry = (f"Pair {i}: First object: '{p['sub_label']}' {sub_box}, "
                 f"Second object: '{p['obj_label']}' {obj_box}")
        pair_entries.append(entry)
    
    pair_text = "\n".join(pair_entries)

    prompt = f"""You are a vision-language expert. Given an image with pairs of objects along with their bounding box coordinates. The bounding box coordinates are defined by (X_top_left, Y_top_left, X_bottom_right, Y_bottom_right) and are scaled between 1 and 1000.

### Object Pair List
{pair_text}

### Output Instructions
- For each pair, write two short sentences:
- Sentence 1: how the first object relates to the second.
- Sentence 2: how the second object relates to the first.
- Focus on spatial or functional interactions.
- Use this format: Pair [index]: Sentence1: | Sentence2:
### Begin:

"""
    return prompt

top_fused = final_top_pairs[:10] # Let's take the top 10

# 2. Get the boxes for these pairs
triplet_candidates = []
for item in top_fused:
    idx = int(item['key'].split('_')[1]) - 1
    triplet_candidates.append(all_info_candidate_pairs[idx])

# 3. Generate the prompt
final_prompt = get_vlm_triplet_prompt(triplet_candidates)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": final_prompt},
        ]
    }
]

# 5. Tokenize and Generate
inputs = processor.apply_chat_template(
    messages, 
    tokenize=True, 
    add_generation_prompt=True, 
    return_dict=True, 
    return_tensors="pt"
).to(model.device)

generated_ids = model.generate(
    **inputs,
    max_new_tokens=1024,
    temperature=0.1,  # Keep temperature low for structured consistency
    repetition_penalty=1.1,
)

# 6. Decode the response
input_len = inputs["input_ids"].shape[1]
response_text = processor.batch_decode(
    generated_ids[:, input_len:], 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=True
)[0]

print("\n--- Model Response ---")
print(response_text)

# 7. Parse the output into a structured format
# This regex looks for: Pair X: Sentence1: ... | Sentence2: ...
parsed_interactions = []
pattern = r"Pair (\d+): Sentence1: (.*?) \| Sentence2: (.*)"

for line in response_text.strip().split('\n'):
    match = re.search(pattern, line)
    if match:
        pair_idx = int(match.group(1))
        sentence1 = match.group(2).strip()
        sentence2 = match.group(3).strip()
        
        parsed_interactions.append({
            "pair_index": pair_idx,
            "interaction_sub_to_obj": sentence1,
            "interaction_obj_to_sub": sentence2
        })

# --- Final Output Check ---
for item in parsed_interactions:
    print(f"\nRelationship for Pair {item['pair_index']}:")
    print(f"{item['interaction_sub_to_obj']}")
