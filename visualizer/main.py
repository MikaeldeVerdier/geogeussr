from metrics_visualizer import MetricsVisualizer
from sample_visualizer import SampleVisualizer

SAVE_PATH = "visualizations"

if __name__ == "__main__":
    """
    # Plot metrics (with model from save_path)
    met_viz = MetricsVisualizer(SAVE_PATH)
    met_viz.plot_metrics(seperate=True, scoped=True, trendline=True, down_sampled_to=1000)
    """

    # """
    # Plot sampling
    sam_viz = SampleVisualizer(SAVE_PATH, "dataset_generator/gadm_410.gpkg")
    # sam_viz = SampleVisualizer(SAVE_PATH, "dissolved_gadm.gpkg", dissolve=False)  # to use un-dissolved (or pre-dissolved)
    sam_viz.plot_sampling()
    # sam_viz.plot_sampling(load_points=True)  # to use saved points from previous visualization
    # """
 