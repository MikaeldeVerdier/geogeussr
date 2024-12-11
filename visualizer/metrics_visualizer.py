import os
import json
import numpy as np
import matplotlib.pyplot as plt

import visualizer_config as viz_cfg

class MetricsVisualizer:
    def __init__(self, load_path, save_path):
        self.save_path = save_path

        self.load_metrics(load_path)

    def load_metrics(self, load_path):
        self.metrics = {}
        for submodel_dir in os.listdir(load_path):
            if submodel_dir.endswith(".keras"):
                continue
            
            metrics_path = os.path.join(load_path, submodel_dir, f"{submodel_dir}_training_log.json")
            if not os.path.exists(metrics_path):
                continue

            with open(metrics_path, "r") as json_file:
                self.metrics[submodel_dir] = json.load(json_file)

    def plot_metrics(self, submodels=None, seperate=False, scoped=False, trendline=False, down_sampled_to=None):
        if submodels is not None:
            chosen_submodels = {submodel: self.metrics[submodel] for submodel in submodels}  # doesn't allow duplicates (creates duplicate images with different inputs)
        else:
            chosen_submodels = self.metrics

        if seperate:
            chosen_submodels = sorted(chosen_submodels.items(), key=lambda submodel: (submodel[0] != "classifier",))  # make classifier be first

            n_rows = int(np.floor(np.sqrt(len(chosen_submodels))))
            n_cols = int(np.ceil(np.sqrt(len(chosen_submodels))))

            if n_rows * n_cols < len(chosen_submodels):
                if n_rows * n_rows >= len(chosen_submodels):
                    n_cols = n_rows
                elif n_cols * n_cols >= len(chosen_submodels):
                    n_rows = n_cols 
        else:
            chosen_submodels = chosen_submodels.items()

            n_rows = 1
            n_cols = 1

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

        for i, (submodel_name, submodel_history) in enumerate(chosen_submodels):
            if seperate:
                row_index = i // n_cols
                col_index = i % n_cols
                ax = axs[row_index, col_index]

                ax.set_title(submodel_name)
            else:
                ax = axs[0, 0]

                if i == 0:
                    ax.set_title(submodels)

            if scoped:
                vals = [v for val in submodel_history.values() for v in val]
                if len(vals) >= 2:
                    median = np.median(vals)
                    median_variance = np.median(np.abs(np.diff(vals)))
                    min_val = np.min(vals)

                    ax.set_ylim(max(min_val - median_variance, 0), median + median_variance)
                    # ax.set_xlim(len(list(submodel_history.values())[0]) - 100, len(list(submodel_history.values())[0]))

            for metric_key, metric_val in submodel_history.items():
                metric_val = np.arange(100000)
                if down_sampled_to != None and len(metric_val) > down_sampled_to:
                    indices = np.linspace(0, len(metric_val), down_sampled_to, dtype=int)
                    used_metric = [
                        np.mean(metric_val[indices[i]:indices[i + 1]])
                        for i in range(down_sampled_to - 1)
                    ]  # length: len(down_sampled_to) - 1

                    used_indices = indices[1:]
                else:
                    used_metric = metric_val
                    used_indices = np.arange(len(used_metric))

                ax.plot(used_indices, used_metric, label=f"{metric_key}{f' ({submodel_name})' if not seperate else ''}")

                if trendline:
                    if len(used_metric) >= 4:  # // 2 > 2
                        cut_off = int(len(used_metric) // 2)
                        x = list(range(cut_off, len(used_metric)))
                        y = used_metric[cut_off:]

                        fit = np.polyfit(x, y, 1)
                        poly = np.poly1d(fit)
                        ax.plot(used_indices[cut_off:], poly(x), label=f"Trendline: {metric_key}{f' ({submodel_name})' if not seperate else ''}", zorder=3)

            ax.set_xscale("linear")
            ax.legend()

            if seperate and i == len(chosen_submodels) - 1:
                for ax in axs[row_index, col_index + 1:]:  # can never be more than n_cols - 1 wrong
                    ax.remove()

        text_vals = [("seperate", seperate), ("scoped", scoped), ("trendline", trendline), (f"down_sampled to {down_sampled_to}", down_sampled_to != None)]
        title = ", ".join([text for text, val in text_vals if val])
        fig.suptitle(f"Metrics for {submodels if submodels is not None else 'all'}{f' ({title})' if title != '' else ''}", wrap=True)
        fig.tight_layout()

        metrics_path = os.path.join(self.save_path, f"metrics_{submodels}su_{seperate}se_{scoped}sc_{trendline}t_{down_sampled_to}d.png")
        plt.savefig(metrics_path)

        plt.close()


if __name__ == "__main__":
    # Plot metrics (with model from save_path)
    met_viz = MetricsVisualizer(viz_cfg.LOAD_PATH, viz_cfg.SAVE_PATH)
    met_viz.plot_metrics(seperate=True, scoped=True, trendline=True, down_sampled_to=1000)
