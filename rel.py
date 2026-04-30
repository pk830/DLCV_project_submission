import json
import re
import ast
import copy

def filter_entry(entry, target_objects, target_relationships):
    """Filters an entry in-place based on target lists."""
    has_valid_objects = False
    has_valid_relationships = False
    
    for conv in entry.get('conversations', []):
        if conv.get('from') == 'human':
            text = conv.get('value', '')
            if text.strip():
                try:
                    parsed_list = ast.literal_eval(f"[{text}]")
                    filtered_list = [d for d in parsed_list if d.get('name') in target_objects]
                    if filtered_list:
                        conv['value'] = ", ".join(repr(d) for d in filtered_list)
                        has_valid_objects = True
                    else:
                        conv['value'] = ""
                except:
                    conv['value'] = ""
                    
        elif conv.get('from') == 'gpt':
            text = conv.get('value', '')
            if text.strip():
                matches = list(re.finditer(r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)', text))
                valid_triplets = []
                for match in matches:
                    subj, rel, obj = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
                    if rel in target_relationships and subj in target_objects and obj in target_objects:
                        valid_triplets.append(match.group(0))
                
                if valid_triplets:
                    conv['value'] = ", ".join(valid_triplets)
                    has_valid_relationships = True
                else:
                    conv['value'] = ""
                    
    return entry if (has_valid_objects or has_valid_relationships) else None

def process_novel_relationships_split(train_file, test_file):
    # 1. Define shared objects (Both sets see all objects)
    shared_objects ={"mountain", "cow", "people", "face", "number", "pizza", "tire", "player", "pillow",
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
    }

    # 2. Split the Relationship Vocabulary (Disjoint Sets)
    # They MUST NOT overlap!
    train_relationships = {"parked on", "growing on", "standing in front of", "wearing", 
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
    "growing in", "sitting in", "cutting", "feeding", "leaning on"
    }


    test_relationships = {"on the front of", "reaching for", "flying", "of",
    "parked along", "talking to", "sitting at", "standing by",
    "hanging on", "covered with", "standing near",
    "full of", "surrounding", "walking in", "reflected in",
    "walking down", "walking on", "contain", "below",
    "printed on", "driving down", "waiting for",
    "resting on", "playing with", "standing in",
    "grazing on", "by", "around", "pulled by", "beneath"}

    print("Loading and pooling datasets...")
    pooled_data = []
    for filepath in [train_file, test_file]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                pooled_data.extend(json.load(f))
        except FileNotFoundError:
            print(f"Warning: {filepath} not found. Skipping.")

    train_split = []
    test_split = []

    print(f"Filtering based on Novel Relationships Split...")
    print(f"Train Relationships ({len(train_relationships)}): {train_relationships}")
    print(f"Test Relationships ({len(test_relationships)}): {test_relationships}")

    for entry in pooled_data:
        # Deep copies to ensure independent filtering
        entry_for_train = copy.deepcopy(entry)
        entry_for_test = copy.deepcopy(entry)

        # Filter for train (shared objects + train relationships)
        train_filtered = filter_entry(entry_for_train, shared_objects, train_relationships)
        if train_filtered:
            train_split.append(train_filtered)

        # Filter for test (shared objects + test relationships)
        test_filtered = filter_entry(entry_for_test, shared_objects, test_relationships)
        if test_filtered:
            test_split.append(test_filtered)

    # Save the results
    print("\nSaving 'novel_rels_train.json' and 'novel_rels_test.json'...")
    with open('novel_rels_train.json', 'w', encoding='utf-8') as f:
        json.dump(train_split, f, indent=2)
    with open('novel_rels_test.json', 'w', encoding='utf-8') as f:
        json.dump(test_split, f, indent=2)
        
    print(f"Done! Train set size: {len(train_split)}. Test set size: {len(test_split)}.")

if __name__ == "__main__":
    process_novel_relationships_split("../gqa_train.json", "../gqa_test.json")