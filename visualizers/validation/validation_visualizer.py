import json
import numpy as np
import matplotlib.pyplot as plt

import validation_viz_config as viz_cfg

class ValidationVisualizer:
    def __init__(self, save_path=viz_cfg.save_path, validation_file=viz_cfg.validation_file):
        self.save_path = save_path

        with open(validation_file, "r") as json_file:
            self.validation_information = json.load(json_file)

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
            # TODO: Is this flawed? Does using plot_density mislead because it is density of correct dists being there, not all dists?
            plt.hist(correct_dists, bins=bins, color="green", label="Correct Region", alpha=0.7)  # Could determine alpha based on confidence
        elif len(incorrect_dists):
            plt.hist(incorrect_dists, bins=bins, color="red", label="Incorrect Region", alpha=0.7)

    def plot_density_histo(self, bins, correct_dists, incorrect_dists):
        correct_hist, _ = np.histogram(correct_dists, bins=bins)
        incorrect_hist, _ = np.histogram(incorrect_dists, bins=bins)

        total_count = len(correct_dists) + len(incorrect_dists)
        bin_width = np.diff(bins)
        correct_hist_density = correct_hist / (total_count * bin_width)
        incorrect_hist_density = incorrect_hist / (total_count * bin_width)

        if len(correct_hist):
            plt.bar(bins[:-1], correct_hist_density, width=bin_width, bottom=incorrect_hist_density if len(incorrect_hist_density) else None, label="Correct Region", color="green", alpha=0.7)
        if len(incorrect_dists):
            plt.bar(bins[:-1], incorrect_hist_density, width=bin_width, label="Incorrect Region", color="red", alpha=0.7)

    def visualize_distribution(self, grouping_size=viz_cfg.grouping_size, plot_density=viz_cfg.plot_density, plot_spaced=viz_cfg.plot_spaced, plot_stacked=viz_cfg.plot_stacked):
        dists = np.array(self.validation_information["distances"])
        was_corrects = np.array(self.validation_information["correct_regions"])

        correct_dists = dists[was_corrects]
        incorrect_dists = dists[~was_corrects]

        # bins = int(len(dists) / grouping_size)  # is this better than just specifying bins?
        bins = np.linspace(min(dists), max(dists), int(len(dists) / grouping_size))
        # if plot_stacked:
        #     self.np_histo(bins, correct_dists, incorrect_dists, plot_density) 
        # else:
        if plot_density:
            self.plot_density_histo(bins, correct_dists, incorrect_dists)
        else:
            self.plt_histo(bins, correct_dists, incorrect_dists, plot_spaced=plot_spaced, plot_stacked=plot_stacked)   
    
        avg_dist = np.mean(dists)
        correct_rate = np.mean(was_corrects)

        plt.title("Distance Distribution for Validation Results")
        plt.suptitle(f"Average Distance: {avg_dist:.2f}km, Correct Rate: {correct_rate * 100:.2f}%")  # a bit weird for this to be...
        plt.xlabel("Distance")
        plt.ylabel("Density" if plot_density else "Count")
        plt.legend()
        plt.tight_layout()

        plt.savefig(self.save_path)

        plt.close()


if __name__ == "__main__":
    visualizer = ValidationVisualizer()
    visualizer.visualize_distribution()
