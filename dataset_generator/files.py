# Don't like this duplication...

import os
import json

def save_annotations(annotations, output_dir):
    with open(os.path.join(output_dir, "_annotations.json"), "w") as json_file:
        json.dump(annotations, json_file)


def load_annotations(input_dir, tolerant=False):
    annotations_path = os.path.join(input_dir, "_annotations.json")

    if tolerant and not os.path.exists(annotations_path):
        return []

    with open(annotations_path, "r") as json_file:
        annotations = json.load(json_file)

    return annotations
