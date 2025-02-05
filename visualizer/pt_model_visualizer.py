import numpy as np
from transformers import CLIPModel, CLIPProcessor
from torchview import draw_graph

import os
import sys
parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_name)  # This sucks!

import visualizer_config as viz_cfg

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
    model_name = "openai/clip-vit-large-patch14-336"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    input_data = processor(["Hello World"], np.random.rand(1, 3, 336, 336), return_tensors="pt", padding=True)
    mod_viz = PTModelVisualizer(model, viz_cfg.SAVE_PATH, input_data)

    mod_viz.visualize(depth=2)
