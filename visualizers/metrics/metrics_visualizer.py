import json
import numpy as np
import matplotlib.pyplot as plt

import metrics_viz_config as viz_cfg

class MetricsVisualizer:
    def __init__(self, save_path=viz_cfg.save_path, metrics_path=viz_cfg.metrics_path):
        self.save_path = save_path
        self.load_metrics(metrics_path)

    def load_metrics(self, metrics_path):
        with open(metrics_path, "r") as json_file:
            self.metrics = json.load(json_file)

        """
        self.metrics = {}
        for submodel_dir in os.listdir(load_path):
            if submodel_dir.endswith(".keras"):
                continue
            
            metrics_path = os.path.join(load_path, submodel_dir, f"{submodel_dir}_training_log.json")
            if not os.path.exists(metrics_path):
                continue

            with open(metrics_path, "r") as json_file:
                self.metrics[submodel_dir] = json.load(json_file)
        """

    def plot_metrics(self, scoped=viz_cfg.scoped, trendline=viz_cfg.trendline, down_sampled_to=viz_cfg.down_sampled_to):
        if scoped:
            vals = [v for val in self.metrics.values() for v in val]
            if len(vals) >= 2:
                median = np.median(vals)
                median_variance = np.median(np.abs(np.diff(vals)))
                min_val = np.min(vals)

                plt.set_ylim(max(min_val - median_variance, 0), median + median_variance)
                # ax.set_xlim(len(list(submodel_history.values())[0]) - 100, len(list(submodel_history.values())[0]))

        for metric_key, metric_val in self.metrics.items():
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

            plt.plot(used_indices, used_metric, label=f"{metric_key}")

            if trendline:
                if len(used_metric) >= 4:  # // 2 > 2
                    cut_off = int(len(used_metric) // 2)
                    x = list(range(cut_off, len(used_metric)))
                    y = used_metric[cut_off:]

                    fit = np.polyfit(x, y, 1)
                    poly = np.poly1d(fit)
                    plt.plot(used_indices[cut_off:], poly(x), label=f"Trendline: {metric_key}", zorder=3)

        plt.title("Metrics During Training")
        plt.legend()

        plt.xlabel("Epochs")
        plt.ylabel("Metric Value")

        # text_vals = [("seperate", seperate), ("scoped", scoped), ("trendline", trendline), (f"down_sampled to {down_sampled_to}", down_sampled_to != None)]
        # title = ", ".join([text for text, val in text_vals if val])
        # fig.suptitle(f"Metrics for {submodels if submodels is not None else 'all'}{f' ({title})' if title != '' else ''}", wrap=True)
        # fig.tight_layout()

        # metrics_path = os.path.join(self.save_path, f"metrics_{submodels}su_{seperate}se_{scoped}sc_{trendline}t_{down_sampled_to}d.png")
        plt.savefig(self.save_path)

        plt.close()


if __name__ == "__main__":
    met_viz = MetricsVisualizer()
    met_viz.plot_metrics()
