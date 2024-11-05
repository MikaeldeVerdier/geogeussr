import os
import json

def load_annotations(input_dir):
    with open(os.path.join(input_dir, "_annotations.json"), "r") as json_file:
        annotations = json.load(json_file)

    return annotations
