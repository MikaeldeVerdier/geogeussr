import os
import json

def save_json(data, path):  # could make this create the path if it doesn't exist
    with open(path, "w") as json_file:
        json.dump(data, json_file, ensure_ascii=False)


def load_json(path):
    with open(path, encoding="utf8") as json_file:
        return json.load(json_file)


def save_annotations(annotations, output_dir):
    save_json(annotations, os.path.join(output_dir, "_annotations.json"))


def load_annotations(input_dir, tolerant=True):
    anno_path = os.path.join(input_dir, "_annotations.json")
    if not os.path.exists(anno_path) and tolerant:
        return []

    annotations = load_json(anno_path)

    return annotations
