import json
import numpy as np
import matplotlib.pyplot as plt

import validation_viz_config as viz_cfg

class ValidationVisualizer:
    def __init__(self, save_path=viz_cfg.save_path, validation_file=viz_cfg.validation_file):
        self.save_path = save_path

        with open(validation_file, "r") as json_file:
            self.validation_information = json.load(json_file)

    """
    def plt_histo(self, bins, correct_dists, incorrect_dists, plot_spaced, plot_stacked):
        if plot_stacked:
            if len(incorrect_dists):
                hist = plt.hist(incorrect_dists, bins=bins, color="red", label="Incorrect Region", alpha=0.7)
            if len(correct_dists):
                plt.hist(correct_dists, bins=bins, bottom=hist[0] if hist else None, color="green", label="Correct Region", alpha=0.7)  # Could determine alpha based on confidence

            return

        if len(correct_dists) and len(incorrect_dists):
            plt.hist([correct_dists, incorrect_dists], rwidth=1 if not plot_spaced else None, bins=bins, color=["green", "red"], label=["Correct Region", "Incorrect Region"], alpha=0.7)
        elif len(correct_dists):
            # Is this flawed? Does using plot_density mislead because it is density of correct dists being there, not all dists?
            plt.hist(correct_dists, bins=bins, color="green", label="Correct Region", alpha=0.7)  # Could determine alpha based on confidence
        elif len(incorrect_dists):
            plt.hist(incorrect_dists, bins=bins, color="red", label="Incorrect Region", alpha=0.7)
        """

    def plot_histo(self, bins, dists, categories, plot_density, category_labels={}, colors={}):
        bin_width = np.diff(bins)
        divisor = len(dists) * bin_width if plot_density else 1

        n_categories_to_plot = np.max(categories) + 1
        category_hists = [np.histogram(dists[categories == i], bins=bins)[0] / divisor for i in range(n_categories_to_plot)]
        for i, (category_hist, color) in enumerate(zip(category_hists, colors)):
            other_category_hists = category_hists[:i]
            tot_other_category_hist = np.sum(other_category_hists, axis=0)

            category_label = category_labels.get(i, f"Category {i}")
            color = colors.get(i, "Black")
            plt.bar(bins[:-1], category_hist, width=bin_width, bottom=tot_other_category_hist, label=category_label, color=color, alpha=0.7)

        # if len(correct_hist):
        #     plt.bar(bins[:-1], correct_hist_density, width=bin_width, bottom=incorrect_hist_density if len(incorrect_hist_density) else None, label="Correct Region", color="green", alpha=0.7)
        # if len(incorrect_dists):
        #     plt.bar(bins[:-1], incorrect_hist_density, width=bin_width, label="Incorrect Region", color="red", alpha=0.7)

    def visualize_distribution(self, grouping_size=viz_cfg.grouping_size, plot_density=viz_cfg.plot_density):
        dists = np.array(self.validation_information["distances"])
        correct_region_levels = np.array(self.validation_information["correct_region_levels"])

        # bins = int(len(dists) / grouping_size)  # is this better than just specifying bins?
        bins = np.linspace(min(dists), max(dists), int(len(dists) / grouping_size))
        # if plot_stacked:
        #     self.np_histo(bins, correct_dists, incorrect_dists, plot_density) 
        # else:

        n_categories = 5
        correct_rates = np.arange(n_categories)[:, None] <= correct_region_levels  # [[res >= i for res in correct_region_levels] for i in range(n_categories)]
        correct_averages = np.sum(correct_rates, axis=-1) / len(correct_region_levels)
        category_labels = {  # cumulative, a bit unintuitive but more informative
            0: f"Nothing Correct ({correct_averages[0] * 100:.2f}%)",
            1: f"Correct continent ({correct_averages[1] * 100:.2f}%)",
            2: f"Correct continent and country ({correct_averages[2] * 100:.2f}%)",
            3: f"Correct continent, country and province ({correct_averages[3] * 100:.2f}%)",
            4: f"Correct continent, country, province and city ({correct_averages[4] * 100:.2f}%)"
        }
        colors = {
            0: "red",
            1: "orange",
            2: "green",
            3: "blue",
            4: "purple"
        }

        self.plot_histo(bins, dists, correct_region_levels, plot_density, category_labels=category_labels, colors=colors)

        avg_dist = np.mean(dists)
        avg_correct_region_level = np.mean(correct_region_levels)

        plt.title("Distance Distribution for Validation Results")
        plt.suptitle(f"Average Distance: {avg_dist:.2f}km, Average Region Correct Level: {avg_correct_region_level}")  # a bit weird for this to be...
        plt.xlabel("Distance")
        plt.ylabel("Density" if plot_density else "Count")
        plt.legend()
        plt.tight_layout()

        plt.savefig(self.save_path)

        plt.close()


if __name__ == "__main__":
    visualizer = ValidationVisualizer()
    visualizer.visualize_distribution()
