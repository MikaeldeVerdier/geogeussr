import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import visualizers.gmm.gmm_viz_config as viz_cfg  # weird for this not to be a __main__ file

class GMMVisualizer:
    def __init__(self):
        self.save_path = viz_cfg.save_path
        self.shapefile_path = viz_cfg.shapefile_path

    def visualize_gmm(self, gm, coords):
        if self.shapefile_path is not None:
            geodf = gpd.read_file(self.shapefile_path)
            geodf = geodf.dissolve(by="GID_0")

            geodf.plot()

        # plt.figure(figsize=(10, 8))
        plt.scatter(coords[:, 1], coords[:, 0], s=10, color="gray", label="Data points")

        def plot_gaussian_ellipse(mean, cov, ax, color):
            eig_vals, eig_vecs = np.linalg.eigh(cov)
            v = 2.0 * np.sqrt(2.0) * np.sqrt(eig_vals)
            u = eig_vecs[1] / np.linalg.norm(eig_vecs[1])
            angle = np.degrees(np.arctan2(u[0], u[1]))

            ellipse = Ellipse((mean[1], mean[0]), v[1], v[0], angle=angle, edgecolor=color, facecolor="none", lw=2)
            ax.add_patch(ellipse)

        for mean, cov in zip(gm.means_, gm.covariances_):
            plot_gaussian_ellipse(mean, cov, plt.gca(), color="red")

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.title("Gaussian Mixture Model Clusters")

        plt.savefig(self.save_path)
        plt.show()
