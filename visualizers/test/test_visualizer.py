import json
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

import test_viz_config as viz_cfg

# TODO: Improve this one to show refinements too (maybe with an animation? Would be really cool)
class TestVisualizer:
    def __init__(self, save_path=viz_cfg.save_path, test_file=viz_cfg.test_file, shapefile_path=viz_cfg.shapefile_path):
        self.save_path = save_path

        with open(test_file, "r") as json_file:
            self.test_information = json.load(json_file)

        self.load_gadm(shapefile_path)

    def load_gadm(self, shapefile_path):
        self.geodf = gpd.read_file(shapefile_path)
        if self.geodf.crs != "EPSG:4326":
            self.geodf = self.geodf.to_crs("EPSG:4326")

    def visualize_test(self):
        fig, axs = plt.subplots(2, 1, figsize=(15, 23), height_ratios=[1, 2])  # figsize would be (15, 20) but a little extra for title space and spacing

        prompt_components = np.array(self.test_information["prompts"])
        image = np.array(self.test_information["image"])
        confs = np.array(self.test_information["confs"])
        best_prompt = np.array(self.test_information["best_prompt"])

        lat_lngs = np.array(prompt_components[:, 1:], dtype=np.float32)

        self.geodf.plot(alpha=0.2, ax=axs[0])
        # norm_confs = confs / np.exp(np.max(confs))
        norm_confs = confs ** 0.75
        axs[0].scatter(lat_lngs[:, 1], lat_lngs[:, 0], c="red", alpha=norm_confs)
        axs[0].scatter(float(best_prompt[2]), float(best_prompt[1]), c="green", alpha=1, label=f"Final Guess ({best_prompt[0]})")
        axs[0].scatter([], [], c="red", alpha=0.5, label="Prompts (intensity based on confidence)")  # just for legend

        text = ""
        conf_order = np.argsort(confs)[::-1][:5]
        for region, conf in zip(prompt_components[conf_order, 0], confs[conf_order]):
            text += f"{region}: {conf * 100:.2f}%\n"

        props = dict(boxstyle="round", facecolor="white", alpha=0.5)
        axs[0].text(0.05, 0.95, text[:-2], transform=axs[0].transAxes, fontsize=14, verticalalignment="top", bbox=props)

        # axs[0].scatter(float(best_prompt[2]), float(best_prompt[1]), c="green", alpha=1)
        axs[0].axis("off")
        axs[0].legend()

        axs[1].imshow(image[..., ::-1])  # shows the image the model used (aka resized to (336x336))
        axs[1].axis("off")

        plt.suptitle("Inference Results for Test Image", fontsize=30)
        plt.tight_layout()

        plt.savefig(self.save_path)

        plt.close()


if __name__ == "__main__":
    visualizer = TestVisualizer()
    visualizer.visualize_test()
