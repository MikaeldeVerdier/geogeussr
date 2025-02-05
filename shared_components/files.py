import os
import json
from functools import lru_cache

def save_json(data, path):
    with open(path, "w") as json_file:
        json.dump(data, json_file, ensure_ascii=False)


def load_json(path):
    with open(path, "r") as json_file:
        return json.load(json_file)


def save_annotations(annotations, output_dir):
    save_json(annotations, os.path.join(output_dir, "_annotations.json"))


@lru_cache()  # loaded for both train_dataset_handler and val_dataset_handler
def load_annotations(input_dir, tolerant=False):
    anno_path = os.path.join(input_dir, "_annotations.json")
    if not os.path.exists(anno_path):
        return []

    annotations = load_json(anno_path)

    return annotations
