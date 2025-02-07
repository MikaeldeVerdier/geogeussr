import os
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from torchview import draw_graph

import model_viz_config as viz_cfg

class PTModelVisualizer:
    def __init__(self, save_path=viz_cfg.save_path, model_name=viz_cfg.model_name, example_prompts=viz_cfg.example_prompts, example_input_shape=viz_cfg.example_input_shape): 
        self.save_path = save_path

        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        example_image = np.random.rand(*example_input_shape)
        input_data = self.processor(example_prompts, example_image, return_tensors="pt", padding=True)
        self.input_data = input_data

    def visualize_model(self, depth=viz_cfg.depth, expand_nested=viz_cfg.expand_nested):
        model_graph = draw_graph(self.model, self.input_data, depth=depth, expand_nested=expand_nested)
        dot = model_graph.visual_graph

        render_file = os.path.join(self.save_path, f"model_{depth}d_{expand_nested}en")
        dot.render(render_file, format="png")


if __name__ == "__main__":
    mod_viz = PTModelVisualizer()
    mod_viz.visualize_model()
