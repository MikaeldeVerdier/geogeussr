import os
import json
from functools import lru_cache

def save_annotations(annotations, output_dir):
    with open(os.path.join(output_dir, "_annotations.json"), "w") as json_file:
        json.dump(annotations, json_file)


@lru_cache()  # loaded for both train_dataset_handler and val_dataset_handler
def load_annotations(input_dir):
    with open(os.path.join(input_dir, "_annotations.json"), "r") as json_file:
        annotations = json.load(json_file)

    return annotations
