import numpy as np
from torchview import draw_graph

import os
import sys
parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_name)  # This sucks!

import visualizer_config as viz_cfg
from model.street_clip.street_clip_model import StreetCLIP
from model.clip_clip.clip_clip_model import ClipCLIP
from model.clip_clip.clip_preprocessor import ClipPreprocessor  # will work for both clip and street

class PTModelVisualizer:
    def __init__(self, model, save_path, input_data):
        self.model = model
        self.save_path = save_path
        self.input_data = input_data

    def visualize(self, depth=3, expand_nested=True):
        model_graph = draw_graph(self.model, input_data, depth=depth, expand_nested=expand_nested)
        dot = model_graph.visual_graph

        render_file = os.path.join(self.save_path, f"model_{depth}d_{expand_nested}en")
        dot.render(render_file, format="png")


if __name__ == "__main__":
    model = ClipCLIP().clip_model
    processor = ClipPreprocessor(None, None).clip_processor  # This sucks too!
    input_data = processor(["Hello World"], np.random.rand(1, 3, 336, 336), return_tensors="pt", padding=True)
    mod_viz = PTModelVisualizer(model, viz_cfg.SAVE_PATH, input_data)

    mod_viz.visualize()
