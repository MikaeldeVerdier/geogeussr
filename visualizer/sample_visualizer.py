import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.ops import nearest_points
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation

import visualizer_config as viz_cfg

class SampleVisualizer:
    def __init__(self, save_path, shapefile_path, dissolve=True, save_dissolved=False):
        self.save_path = save_path

        self.geodf = gpd.read_file(shapefile_path)
        if dissolve:
            self.geodf = self.geodf.dissolve()
        if self.geodf.crs != "EPSG:4326":
            self.geodf = self.geodf.to_crs("EPSG:4326")
        if save_dissolved:
            self.geodf.to_file("dissolved_gadm.gpkg", driver="GPKG")

    """
    def get_points(self, df):
        points = []
        for geom in df.geometry:
            if geom.geom_type == "Point":
                points.append(geom.coords[0])
            elif geom.geom_type in ["LineString", "Polygon"]:
                points.extend(list(geom.coords))
            elif geom.geom_type in ["MultiPoint", "MultiLineString", "MultiPolygon"]:
                for part in geom:
                    points.extend(list(part.coords))

        return points
    """

    def generate_heatmap(self, points, s, b):
        heatmap, x_edges, y_edges = np.histogram2d(points[:, 0], points[:, 1], bins=b)  # doesn't histo across world's edge
        heatmap = gaussian_filter(heatmap, sigma=s)

        # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        return heatmap.T

    def plot_sampling(self, points, file_name, smoothing, bins, normalize_heatmap, show_map, show_points):
        if show_map:
            ax = self.geodf.plot(alpha=0.2)

            bounds = self.geodf.total_bounds  # NOT [-180, 180, -90, 90] (for some reason doesn't go up to north pole)
            extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
        else:
            ax = plt.gca()  # could just plot imshow first...
            extent = [-180, 180, -90, 90]

        heat_map = self.generate_heatmap(points, smoothing, bins)
        if normalize_heatmap:  # Could use a log_normalizer instead to get landmasses some color. Don't think it's worth though because coast lines should be this emphasized
            v_min = np.min(heat_map)  # will almost assuredly be 0
            v_max = np.max(heat_map)
        else:
            v_min = 0
            v_max = 1
        cmap = LinearSegmentedColormap.from_list("CustomHeat", [
            (0, "black"),
            (0.001, "green"),
            (0.1, "yellow"),
            (1.0, "red")
        ])
        # cmap = "rainbow"

        heat_map_img = ax.imshow(heat_map, extent=extent, origin="lower", cmap=cmap, vmin=v_min, vmax=v_max)
        cbar = plt.colorbar(heat_map_img, location="top")
        cbar.set_label("Sample point density (points per pixel)", rotation=0, labelpad=15)

        if show_points:
            ax.scatter(points[:, 0], points[:, 1], color="r", s=1, zorder=4)

        plt.axis("off")

        sampling_path = os.path.join(self.save_path, file_name)
        plt.savefig(sampling_path, dpi=1000, bbox_inches="tight")

        plt.close()

    def plot_refined_sampling(self, load_points=False, n_points=1000000, smoothing=0, bins=300, normalize_heatmap=True, show_map=True, show_points=False):
        points_path = os.path.join(self.save_path, f"{n_points}_refined_points.npy")
        if not load_points:
            points = self.geodf.sample_points(n_points)
            used_points = np.array([geom.coords[0] for geom in points.iloc[0].geoms])

            np.save(points_path, used_points)
        else:
            used_points = np.load(points_path)

        image_path = f"sampling_refined_{n_points}np_{smoothing}smo_{bins}b_{normalize_heatmap}no_{show_map}shm_{show_points}shp.png"
        self.plot_sampling(used_points, image_path, smoothing, bins, normalize_heatmap, show_map, show_points)

    def simulate_sampling(self, points, domain_points):
        used_points = []
        for point in points:
            on_land = np.any(self.geodf.contains(point))
            if on_land:
                used_points.append([point.x, point.y])

                continue

            nearest = nearest_points(point, domain_points)[1].values[0]
            used_points.append([nearest.x, nearest.y])

        return np.array(used_points)

    def plot_naive_sampling(self, load_points=False, n_points=1000, smoothing=0, bins=300, normalize_heatmap=True, show_map=True, show_points=False, animate=False):
        if not load_points or animate:  # ugly but
            n_lats = round(np.sqrt(n_points * 2 / 3))
            n_lngs = round(n_points / n_lats)

            lats = np.linspace(-90, 90, n_lats)
            lngs = np.linspace(-180, 180, n_lngs)
            np_points = np.array([[lng, lat] for lat in lats for lng in lngs])

        points_path = os.path.join(self.save_path, f"{n_points}_naive_points.npy")
        if not load_points:  # Why haven't I looked into Hammersley sampling?
            shp_points = gpd.points_from_xy(np_points[:, 0], np_points[:, 1])
            used_points = self.simulate_sampling(tuple(shp_points), self.geodf.boundary)
            # boundary.plot()
            # boundary_points = gpd.GeoDataFrame(geometry=self.get_points(boundary), crs=boundary.crs)
        
            np.save(points_path, used_points)
        else:
            used_points = np.load(points_path)

        if animate:
            animation_path = f"ani_sampling_refined_{n_points}np_{smoothing}smo_{bins}b_{normalize_heatmap}no_{show_map}shm_{show_points}shp.mp4"
            self.animate_sampling(np_points, used_points, animation_path)
        else:
            image_path = f"sampling_naive_{n_points}np_{smoothing}smo_{bins}b_{normalize_heatmap}no_{show_map}shm_{show_points}shp.png"
            self.plot_sampling(used_points, image_path, smoothing, bins, normalize_heatmap, show_map, show_points)

    def animate_sampling(self, original_points, used_points, file_name, num_frames=100):
        ax = self.geodf.plot(alpha=0.2)

        scat = ax.scatter(original_points[:, 0], used_points[:, 1], color="r", s=1)

        def interpolate_points(start, end, alpha):
            return start + alpha * (end - start)

        def update(frame):
            alpha = frame / num_frames
            current_points = interpolate_points(original_points, used_points, alpha)
            scat.set_offsets(current_points)

            return scat

        fig = plt.gcf()
        ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

        plt.axis("off")

        # plt.show()
        animation_path = os.path.join(self.save_path, file_name)
        ani.save(animation_path, writer="ffmpeg", fps=40)


if __name__ == "__main__":
    # Plot sampling
    sam_viz = SampleVisualizer(viz_cfg.SAVE_PATH, "dataset_generator/gadm_410.gpkg")
    # sam_viz = SampleVisualizer(viz_cfg.SAVE_PATH, "dissolved_gadm.gpkg", dissolve=False)  # to use un-dissolved (or pre-dissolved)
    # sam_viz.plot_naive_sampling()  # Using naive sampling
    # sam_viz.plot_naive_sampling(n_points=1000)  # To animate naive sampling (lower n_points recommended)
    sam_viz.plot_refined_sampling()  # Using refined sampling
    # sam_viz.plot_refined_sampling(load_points=True)  # to use saved points from previous visualization
