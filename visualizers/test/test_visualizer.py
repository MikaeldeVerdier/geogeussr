import json
import numpy as np
import matplotlib.pyplot as plt

import test_viz_config as viz_cfg

class TestVisualizer:
    def __init__(self, save_path=viz_cfg.save_path, test_file=viz_cfg.test_file):
        self.save_path = save_path

        with open(test_file, "r") as json_file:
            self.test_information = json.load(json_file)

    def visualize_distribution(self, grouping_size=viz_cfg.grouping_size, plot_density=viz_cfg.plot_density):
        dists = np.array(self.test_information["distances"])
        was_corrects = np.array(self.test_information["correct_regions"])

        correct_dists = dists[was_corrects]
        incorrect_dists = dists[~was_corrects]

        bins = int(len(dists) / grouping_size)  # is this better than just specifying bins?
        plt.hist([correct_dists, incorrect_dists], bins=bins, density=plot_density, color=["green", "red"], label=["Correct Region", "Incorrect Region"], alpha=0.7)

        avg_dist = np.mean(dists)
        correct_rate = np.mean(was_corrects)

        plt.title("Distance Distribution for Testing Results")
        plt.suptitle(f"Average Distance: {avg_dist:.2f}km, Correct Rate: {correct_rate * 100:.2f}%")
        plt.xlabel("Distance")
        plt.ylabel("Density" if plot_density else "Count")
        plt.legend()
        plt.tight_layout()

        plt.savefig(self.save_path)


if __name__ == "__main__":
    visualizer = TestVisualizer()
    visualizer.visualize_distribution()
