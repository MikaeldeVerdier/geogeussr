import numpy as np
import pickle as pkl
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import gmm_viz_config as viz_cfg  # weird for this not to be a __main__ file

class GMMVisualizer:
    def __init__(self, save_path=viz_cfg.save_path, shapefile_path=viz_cfg.shapefile_path, gmm_path=viz_cfg.gmm_path):
        self.save_path = save_path

        self.load_geodf(shapefile_path)
        self.load_gmm(gmm_path)

    def load_geodf(self, path):
        self.geodf = gpd.read_file(path)
        self.geodf = self.geodf.dissolve(by="GID_0")

    def load_gmm(self, path):
        with open(path, "rb") as pkl_file:
            self.gm = pkl.load(pkl_file)

    def visualize_gmm(self, coords=viz_cfg.coords):
        # plt.figure(figsize=(10, 8))
        plt.scatter(coords[:, 1], coords[:, 0], s=10, color="gray", label="Data points")

        def plot_gaussian_ellipse(mean, cov, ax, color):
            eig_vals, eig_vecs = np.linalg.eigh(cov)
            v = 2.0 * np.sqrt(2.0) * np.sqrt(eig_vals)
            u = eig_vecs[1] / np.linalg.norm(eig_vecs[1])
            angle = np.degrees(np.arctan2(u[0], u[1]))

            ellipse = Ellipse((mean[1], mean[0]), v[1], v[0], angle=angle, edgecolor=color, facecolor="none", lw=2)
            ax.add_patch(ellipse)

        for mean, cov in zip(self.gm.means_, self.gm.covariances_):
            plot_gaussian_ellipse(mean, cov, plt.gca(), color="red")

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.title("Gaussian Mixture Model Clusters")

        plt.savefig(self.save_path)
        plt.show()


if __name__ == "__main__":
    viz = GMMVisualizer()
    viz.visualize_gmm()
