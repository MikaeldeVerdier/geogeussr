from metrics_visualizer import MetricsVisualizer

SAVE_PATH = "save_folder"

if __name__ == "__main__":
    # Plot metrics (from save_path)
    # """
    viz = MetricsVisualizer(SAVE_PATH)
    viz.plot_metrics(["classifier", "GBR"], seperate=True, scoped=True, trendline=True)
    # """
