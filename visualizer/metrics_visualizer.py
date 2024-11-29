import os
import json
import numpy as np
import matplotlib.pyplot as plt

class MetricsVisualizer:
    def __init__(self, save_path):
        self.save_path = save_path

        self.load_metrics(save_path)

    def load_metrics(self, save_path):
        self.metrics = {}
        for submodel_dir in os.listdir(save_path):
            if submodel_dir.endswith(".keras"):
                continue
            
            metrics_path = os.path.join(save_path, submodel_dir, f"{submodel_dir}_training_log.json")
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
                metric_val = np.random.rand(1000)
                if down_sampled_to != None and len(metric_val) > down_sampled_to:
                    block_size = len(metric_val) // down_sampled_to
                    used_metric = [
                        np.mean(metric_val[block_size * i: block_size * (i + 1)])
                        for i in range(down_sampled_to)
                    ]
                else:
                    used_metric = metric_val

                ax.plot(used_metric, label=f"{metric_key}{f' ({submodel_name})' if not seperate else ''}")

                if trendline:
                    if len(used_metric) >= 4:  # // 2 > 2
                        cut_off = int(len(used_metric) // 2)
                        x = list(range(cut_off, len(used_metric)))
                        y = used_metric[cut_off:]

                        fit = np.polyfit(x, y, 1)
                        poly = np.poly1d(fit)
                        ax.plot(x, poly(x), label=f"Trendline: {metric_key}{f' ({submodel_name})' if not seperate else ''}", zorder=3)

            ax.set_xscale("linear")
            ax.legend()

            if seperate and i == len(chosen_submodels) - 1:
                for ax in axs[row_index, col_index + 1:]:  # can never be more than n_cols - 1 wrong
                    ax.remove()

        title = ", ".join([text for text, val in [("seperate", seperate), ("scoped", scoped), ("trendline", trendline), (f"down_sampled to {down_sampled_to}", down_sampled_to != None)] if val])
        fig.suptitle(f"Metrics for {submodels if submodels is not None else 'all'}{f' ({title})' if title != '' else ''}", wrap=True)
        fig.tight_layout()

        metrics_path = os.path.join(self.save_path, f"metrics_{submodels}su_{seperate}se_{scoped}sc_{trendline}t_{down_sampled_to}d.png")
        plt.savefig(metrics_path)
