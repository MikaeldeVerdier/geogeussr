import os
import json
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

import inference_viz_config as viz_cfg

# TODO: Improve this one to show refinements too (maybe with an animation? Would be really cool)
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

    def plot_scatter(self, ax, lat_lngs, confs, prompt_components, best_prompt, correct_prompt):
        self.geodf.plot(alpha=0.2, ax=ax)
        # norm_confs = confs / np.exp(np.max(confs))
        norm_confs = confs ** 0.75
        ax.scatter(lat_lngs[:, 1], lat_lngs[:, 0], c="red", alpha=norm_confs)
        ax.scatter(best_prompt[2], best_prompt[1], c="blue", alpha=1, label=f"Final Guess ({best_prompt[0]})")
        if correct_prompt is not None:
            ax.scatter(correct_prompt[2], correct_prompt[1], c="green", alpha=1, label=f"Correct Answer ({correct_prompt[0]})")
        ax.scatter([], [], c="red", alpha=0.5, label="Prompts (intensity based on confidence)")  # just for legend

        text = ""
        conf_order = np.argsort(confs)[::-1][:5]
        for region, conf in zip(prompt_components[conf_order, 0], confs[conf_order]):
            text += f"{region}: {conf * 100:.2f}%\n"

        props = dict(boxstyle="round", facecolor="white", alpha=0.5)
        ax.text(0.05, 0.95, text[:-2], transform=ax.transAxes, fontsize=15, verticalalignment="top", bbox=props)

        # ax.scatter(float(best_prompt[2]), float(best_prompt[1]), c="green", alpha=1)
        ax.set_title("Results", fontsize=20)
        ax.axis("off")
        ax.legend(loc="upper right", fontsize=15)

    def visualize_inferences(self):
        for inference_name, inference_information in self.inference_informations.items():
            fig, axs = plt.subplots(1, 2, figsize=(21, 9), width_ratios=[1, 2])  # figsize would be (21, 7) but a little extra for title space and spacing

            prompt_components = np.array(inference_information["prompts"])
            image = np.array(inference_information["image"])
            confs = np.array(inference_information["confs"])
            best_prompt = inference_information["best_prompt"]
            correct_prompt = inference_information.get("correct_prompt", None)

            lat_lngs = np.array(prompt_components[:, 1:], dtype=np.float32)

            self.plot_image(axs[0], image)
            self.plot_scatter(axs[1], lat_lngs, confs, prompt_components, best_prompt, correct_prompt)

            plt.suptitle("Inference Results for Image", fontsize=40)
            plt.tight_layout()

            inference_path = os.path.join(self.save_path, f"{inference_name}.png")
            plt.savefig(inference_path)

            plt.close()


if __name__ == "__main__":
    visualizer = InferenceVisualizer()
    visualizer.visualize_inferences()
