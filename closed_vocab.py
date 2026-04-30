import json
import re
import ast
import copy
import pdb

import ast
import re
import random

import re
import ast
import random

def filter_entry_train(entry, 
                       target_objects, 
                       shared_relationships, 
                       novel2base=None, 
                       keep_base_objects=False):
    """
    Training filter with novel→base mapping.
    Handles your specific messy human message format.
    """
    if novel2base is None:
        novel2base = {}

    has_valid_objects = False
    has_valid_relationships = False
    

    for conv in entry.get('conversations', []):
        
        # ====================== HUMAN PART: Extract Objects ======================
        if conv.get('from') == 'human':
            text = conv.get('value', '')
            if not text:
                continue

            try:
                # Extract all {'name': '...', 'bbox': [...]} patterns using regex
                # This is more reliable than ast.literal_eval for your format
                pattern = r"\{'name':\s*'([^']+)'[^}]*'bbox':\s*\[([^\]]+)\]\}"
                matches = re.findall(pattern, text)
                
                filtered_list = []
                
                for name, bbox_str in matches:
                    if not name:
                        continue
                        
                    final_name = name.strip()
                    
                    # Apply novel → base mapping if enabled
                    if (keep_base_objects and 
                        final_name in novel2base and 
                        novel2base[final_name]):
                        final_name = random.choice(novel2base[final_name])
                    
                    # Keep only if final name is allowed
                    if final_name in target_objects:
                        # Reconstruct dict
                        new_d = {
                            'name': final_name,
                            'bbox': [int(x.strip()) for x in bbox_str.split(',')]
                        }
                        filtered_list.append(new_d)
                        has_valid_objects = True
                
                # Update human message
                if filtered_list:
                    conv['value'] = ", ".join(repr(d) for d in filtered_list)
                else:
                    conv['value'] = ""
                    
            except Exception as e:
                print(f"Warning: Failed to parse objects: {e}")
                conv['value'] = ""

        # ====================== GPT PART: Relationships ======================
        elif conv.get('from') == 'gpt':
            #pdb.set_trace()
            text = conv.get('value', '')
            if not text:
                continue
                
            matches = list(re.finditer(r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)', text))
            valid_triplets = []
            
            for match in matches:
                subj = match.group(1).strip()
                rel = match.group(2).strip()
                obj = match.group(3).strip()
                
                # Apply novel→base mapping
                mapped_subj = subj
                mapped_obj = obj
                
                if keep_base_objects:
                    if subj in novel2base and novel2base[subj]:
                        mapped_subj = random.choice(novel2base[subj])
                    if obj in novel2base and novel2base[obj]:
                        mapped_obj = random.choice(novel2base[obj])
                
                # Strict check: relation must be shared + both objects must be allowed
                if (rel in shared_relationships and 
                    mapped_subj in target_objects and 
                    mapped_obj in target_objects):
                    
                    if mapped_subj != subj or mapped_obj != obj:
                        new_triplet = f"({mapped_subj}, {rel}, {mapped_obj})"
                        valid_triplets.append(new_triplet)
                    else:
                        valid_triplets.append(match.group(0))
                    
                    has_valid_relationships = True
            
            if valid_triplets:
                conv['value'] = ", ".join(valid_triplets)
            else:
                conv['value'] = ""

    return entry if (has_valid_objects or has_valid_relationships) else None

def filter_entry_test(entry, all_objects, shared_relationships):
    """
    Strict filter for test/validation set.
    
    Rules:
    - All objects in triplets must be in `all_objects`
    - All relations in triplets must be in `shared_relationships`
    - No novel-to-base mapping is applied (this is for test time)
    - Keeps the entry only if at least one valid triplet remains
    
    This is useful when you want to evaluate only on "shared" / controlled 
    object-relation combinations.
    """
    has_valid_objects = False
    has_valid_relationships = False

    for conv in entry.get('conversations', []):
        
        # ====================== HUMAN PART: Objects ======================
        if conv.get('from') == 'human':
            text = conv.get('value', '')
            if not text.strip():
                continue
                
            try:
                # Robust object extraction using regex (works with your messy format)
                # Pattern matches: {'name': 'banana', 'bbox': [248, 55, 64, 34]}
                pattern = r"\{'name':\s*'([^']+)'[^}]*'bbox':\s*\[([^\]]+)\]\}"
                matches = re.findall(pattern, text)
                
                filtered_list = []
                
                for name, bbox_str in matches:
                    name = name.strip()
                    if not name:
                        continue
                        
                    # Keep only if the object is in all_objects (for test mode)
                    if name in all_objects:
                        # Reconstruct clean dictionary
                        try:
                            bbox = [int(x.strip()) for x in bbox_str.split(',')]
                            new_d = {'name': name, 'bbox': bbox}
                            filtered_list.append(new_d)
                        except:
                            # If bbox parsing fails, still keep the name
                            new_d = {'name': name}
                            filtered_list.append(new_d)
                
                # Update the conversation value
                if filtered_list:
                    conv['value'] = ", ".join(repr(d) for d in filtered_list)
                    has_valid_objects = True
                else:
                    conv['value'] = ""
                    
            except Exception as e:
                print(f"Warning: Failed to parse human objects: {e}")
                conv['value'] = ""

        # ====================== GPT PART: Relationships (Strict) ======================
        elif conv.get('from') == 'gpt':
            text = conv.get('value', '')
            if not text.strip():
                continue
                
            # Extract all triplets: (subject, relation, object)
            matches = list(re.finditer(r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)', text))
            valid_triplets = []
            
            for match in matches:
                subj = match.group(1).strip()
                rel  = match.group(2).strip()
                obj  = match.group(3).strip()
                
                # STRICT CONDITION:
                # Both objects must be in all_objects AND relation must be in shared_relationships
                if (subj in all_objects and 
                    obj in all_objects and 
                    rel in shared_relationships):
                    
                    valid_triplets.append(match.group(0))
                    has_valid_relationships = True
            
            # Update with only valid triplets
            if valid_triplets:
                conv['value'] = ", ".join(valid_triplets)
            else:
                conv['value'] = ""

    # Keep entry only if it has valid objects OR valid relationships
    return entry if (has_valid_objects or has_valid_relationships) else None

def process_novel_objects_split(train_file, test_file):
    # 1. Define shared relationships
    shared_relationships = {
    "parked on", "growing on", "standing in front of", "wearing", 
    "standing on", "with", "looking at", "under", "carrying", "near", 
    "above", "covered in", "behind", "at", "using", "hanging from", 
    "sitting on", "flying in", "watching", "covering", "mounted on", 
    "in front of", "lying on", "standing next to", "grazing in", 
    "holding", "beside", "on the back of", "catching", "running on", 
    "swimming in", "playing on", "on top of", "floating in", 
    "talking on", "on the bottom of", "standing behind", 
    "leaning against", "covered by", "facing", "filled with", 
    "attached to", "sitting next to", "next to", "worn on", "in", 
    "on the side of", "driving", "close to", "surrounded by", 
    "lying in", "hitting", "pulling", "swinging", "touching", 
    "eating", "throwing", "skiing on", "driving on", 
    "riding", "playing in", "crossing", "walking with", "on", 
    "growing in", "sitting in", "cutting", "feeding", "leaning on",
    "on the front of", "reaching for", "flying", "of",
    "parked along", "talking to", "sitting at", "standing by",
    "hanging on", "covered with", "standing near",
    "full of", "surrounding", "walking in", "reflected in",
    "walking down", "walking on", "contain", "below",
    "printed on", "driving down", "waiting for",
    "resting on", "playing with", "standing in",
    "grazing on", "by", "around", "pulled by", "beneath", "to the left of", "to the right of"
    }

    # 2. Split the Object Vocabulary (Disjoint Sets)
    # You can change which objects go where, but they MUST NOT overlap!
    train_objects = {"mountain", "cow", "people", "face", "number", "pizza", "tire", "player", "pillow",
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
    "arrow", "field", "bread", "bicycle", "knife", "couch", "ceiling"}

    novel_objects = {"ocean", "car", "picture", "hand", "snow", "horn", "woman", "sweater", "container",
    "paint", "feet", "clouds", "foot", "dirt", "faucet", "chair", "sand", "tail", "stone", "cat",
    "tag", "traffic light", "keyboard", "tree", "leaves", "elephant", "ground", "glass",
    "frisbee", "trash can", "word", "man", "jeans", "door", "building", "sky", "table",
    "wheel", "pole", "collar", "hat", "cheese", "mane", "shirt", "dog", "cord", "cake",
    "donut", "plate", "backpack", "mirror", "street light", "skis", "window", "grass",
    "water", "bike", "road", "head", "cell phone"}

    all_objects = train_objects | novel_objects

    novel_2_base = {
    # Strong semantic mappings (multiple choices where reasonable)
    "car": ["vehicle"],
    "woman": ["person", "lady"],
    "man": ["person", "guy"],
    "stone": ["rock"],
    "tree": ["plant"],
    "leaves": ["leaf"],
    "elephant": ["animal"],
    "cat": ["animal"],
    "dog": ["animal"],
    "jeans": ["pants"],
    "shirt": ["t-shirt", "jacket", "coat"],
    "wheel": ["tire"],
    "pole": ["post"],
    "cord": ["wire"],
    "backpack": ["bag"],
    "bike": ["bicycle"],
    "road": ["street"],
    "cell phone": ["phone"],
    "cheese": ["food"],
    "cake": ["food"],
    "donut": ["food"],

    # Reasonable but slightly weaker mappings
    "dirt": ["ground"],
    "sand": ["ground", "beach"],
    "glass": ["cup", "bottle"],
    "table": ["desk", "counter"],
    "building": ["house"],
    "tag": ["sticker", "logo"],
    "street light": ["lamp", "post"],
    "grass": [],
    "mane": ["hair", "fur"],
    "container": ["box", "bottle"],

    # No good semantic match in your base list → map to None or empty list
    "ocean": [],
    "picture": [],
    "hand": [],
    "snow": [],
    "horn": [],
    "sweater": [],
    "paint": [],
    "feet": [],
    "clouds": [],
    "foot": [],
    "faucet": [],
    "chair": [],
    "tail": [],
    "traffic light": [],
    "keyboard": [],
    "ground": ["ground"],          # already in base, but kept for completeness
    "frisbee": [],
    "trash can": [],
    "word": [],
    "door": [],
    "sky": [],
    "collar": [],
    "hat": [],
    "mirror": [],
    "skis": [],
    "window": [],
    "water": [],
    "head": [],
    "plate": []
}
    print("Loading and pooling datasets...")
    
    pooled_train_data = []
    
    for filepath in [train_file]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                pooled_train_data.extend(json.load(f))
        except FileNotFoundError:
            print(f"Warning: {filepath} not found. Skipping.")

    train_split = []
    print(f"Filtering based on Novel Objects Split...")
    print(f"Train Objects ({len(train_objects)}): {train_objects}")
    #print(f"Test Objects ({len(novel_objects)}): {novel_objects}")
    pdb.set_trace()
    for entry in pooled_train_data:
        entry_for_train = copy.deepcopy(entry)
        train_filtered = filter_entry_train(
                                    entry=entry,
                                    target_objects=train_objects,
                                    shared_relationships=shared_relationships,
                                    novel2base=novel_2_base,
                                    keep_base_objects=True
                                )
        if train_filtered:
            train_split.append(train_filtered)
    cleaned_train_split = []

    for entry in train_split:
        # Check if any GPT conversation has non-empty value
        has_valid_gpt = False
        
        for conv in entry.get('conversations', []):
            if conv.get('from') == 'gpt':
                gpt_value = conv.get('value', '').strip()
                if gpt_value:                    # if not empty string
                    has_valid_gpt = True
                    break
        
        # Keep the entry only if it has at least one non-empty GPT response
        if has_valid_gpt:
            cleaned_train_split.append(entry)

    # Now save the cleaned version
    with open('novel_objects_train.json', 'w', encoding='utf-8') as f:
        json.dump(cleaned_train_split, f, indent=2)

    print(f"Original entries: {len(train_split)}")
    print(f"After removing empty GPT entries: {len(cleaned_train_split)}")
    
    
    
    pooled_test_data = []
    for filepath in [test_file]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                pooled_test_data.extend(json.load(f))
        except FileNotFoundError:
            print(f"Warning: {filepath} not found. Skipping.")
    test_split = []
    for entry in pooled_test_data:
        # Filter for train (only keeps train objects)
        test_filtered = filter_entry_test(
                                    entry=entry,
                                    all_objects=all_objects,        # usually base + novel
                                    shared_relationships=shared_relationships # common relations you want to evaluate on
                                )
        if test_filtered:
            test_split.append(test_filtered)
    
    cleaned_test_split = []
    for entry in test_split:
        has_valid_gpt = False
        
        for conv in entry.get('conversations', []):
            if conv.get('from') == 'gpt':
                gpt_value = conv.get('value', '').strip()
                if gpt_value:                    # If GPT has content
                    has_valid_gpt = True
                    break
        
        # Keep entry only if it has at least one non-empty GPT response
        if has_valid_gpt:
            cleaned_test_split.append(entry)

    # Save the cleaned test split
    with open('novel_objects_test.json', 'w', encoding='utf-8') as f:
        json.dump(cleaned_test_split, f, indent=2)

    print(f"Original test entries: {len(test_split)}")
    print(f"Final test entries after removing empty GPT: {len(cleaned_test_split)}")
        
    print(f"Done! Train set size: {len(train_split)}. Test set size: {len(test_split)}.")

if __name__ == "__main__":
    process_novel_objects_split("/home/venky/koushikpavan/data_ovsgg/gqa_bb_listed/gqa_structured_train.json", "/home/venky/koushikpavan/data_ovsgg/gqa_bb_listed/gqa_structured_test.json")