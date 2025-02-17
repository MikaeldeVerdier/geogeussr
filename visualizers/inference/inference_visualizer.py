import os
import json
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import inference_viz_config as viz_cfg

class InferenceVisualizer:
    def __init__(self, save_path=viz_cfg.save_path, inference_dir=viz_cfg.inference_dir, shapefile_path=viz_cfg.shapefile_path):
        self.save_path = save_path

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        self.load_inferences(inference_dir)
        self.load_gadm(shapefile_path)

    def load_inferences(self, inference_dir):
        self.inference_informations = {}
        for inference_file in os.listdir(inference_dir):
            image_name = inference_file.replace(".json", "")

            inference_path = os.path.join(inference_dir, inference_file)
            with open(inference_path, "r") as json_file:
                self.inference_informations[image_name] = json.load(json_file)

    def load_gadm(self, shapefile_path):
        self.geodf = gpd.read_file(shapefile_path)
        if self.geodf.crs != "EPSG:4326":
            self.geodf = self.geodf.to_crs("EPSG:4326")

    def plot_image(self, ax, image):
        ax.imshow(image)  # shows the image the model used (aka resized to (336x336))
        ax.set_title("Image", fontsize=20)
        ax.axis("off")

    def get_prompt_description(self, prompt_components):
        if len(prompt_components) == 0:
            return ""

        if len(prompt_components) == 1:
            return prompt_components[0]

        if len(prompt_components) == 2:
            return prompt_components[1]

        if len(prompt_components) == 3:
            return f"{prompt_components[2]}, {prompt_components[1]}"

        if len(prompt_components) == 4:
            if prompt_components[3]:
                return f"{prompt_components[3]}, {prompt_components[1]}"
            else:
                return f"Rural {prompt_components[2]}, {prompt_components[1]}"

    def plot_scatter(self, ax, lat_lngs, confs, prompt_components, best_prompt, correct_prompt, refinement_idx=None):
        self.geodf.plot(alpha=0.2, ax=ax)
        # norm_confs = confs / np.exp(np.max(confs))
        norm_confs = confs ** 0.75
        ax.scatter(lat_lngs[:, 1], lat_lngs[:, 0], c="red", alpha=norm_confs)
        ax.scatter(best_prompt[1][1], best_prompt[1][0], c="blue", alpha=1, label=f"Final Guess ({self.get_prompt_description(best_prompt[0])})")
        if correct_prompt is not None:
            ax.scatter(correct_prompt[1][1], correct_prompt[1][0], c="green", alpha=1, label=f"Correct Answer ({self.get_prompt_description(correct_prompt[0])})")
        ax.scatter([], [], c="red", alpha=0.5, label="Prompts (intensity based on confidence)")  # just for legend

        text = ""
        conf_order = np.argsort(confs)[::-1][:5]
        regions = [prompt_components[idx][0] for idx in conf_order]
        for region, conf in zip(regions, confs[conf_order]):
            text += f"{self.get_prompt_description(region)}: {conf * 100:.2f}%\n"

        props = dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=(0.8, 0.8, 0.8), alpha=0.5)
        ax.text(0.015, 1.2, text[:-1], transform=ax.transAxes, fontsize=15, verticalalignment="top", bbox=props)  # estimated to imitate legend

        # ax.scatter(float(best_prompt[2]), float(best_prompt[1]), c="green", alpha=1)
        if refinement_idx is not None:
            ax.set_title(f"Results\n(refinement level: {refinement_idx + 1})", fontsize=20)
        else:
            ax.set_title("Results", fontsize=20)
        # ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="lower right", fontsize=15, bbox_to_anchor=(1, 1), framealpha=0.5)  # would like to pass props to here to unify

    def visualize_inferences(self):
        for inference_name, inference_information in self.inference_informations.items():
            image = np.array(inference_information["image"])
            correct_prompt = inference_information.get("correct_prompt", None)
            refinement_results = inference_information["refinement_results"]
            # for refinement_idx, refinement_result in enumerate(refinement_results):

            fig, axs = plt.subplots(1, 2, figsize=(21, 11), width_ratios=[1, 2])  # figsize would be (21, 7) but a little extra for title space and spacing

            self.plot_image(axs[0], image)

            # TODO: Consider zooming the image to see the prompts better for deeper refinements
            def update_frame(frame):
                axs[1].clear()

                prompt_components = refinement_results[frame]["prompt_components"]
                best_prompt = refinement_results[frame]["used_prompt_components"]
                confs = np.array(refinement_results[frame]["confidences"])

                lat_lngs = np.array([components[1] for components in prompt_components])

                self.plot_scatter(axs[1], lat_lngs, confs, prompt_components, best_prompt, correct_prompt, refinement_idx=frame)

            # consider saving an image if only one refinement level or maybe even allowing individual frame saving
            ani = FuncAnimation(fig, update_frame, frames=len(refinement_results), interval=1500)
            plt.suptitle("Inference Results for Image", fontsize=40)
            plt.tight_layout()

            inference_path = os.path.join(self.save_path, f"{inference_name}.mp4")
            ani.save(inference_path, writer="ffmpeg")


if __name__ == "__main__":
    visualizer = InferenceVisualizer()
    visualizer.visualize_inferences()
