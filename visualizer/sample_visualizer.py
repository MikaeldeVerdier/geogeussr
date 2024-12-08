import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.ops import nearest_points
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap

class SampleVisualizer:
    def __init__(self, save_path, shapefile_path, dissolve=True):
        self.save_path = save_path

        self.geodf = gpd.read_file(shapefile_path)
        if dissolve:
            self.geodf = self.geodf.dissolve()
        if self.geodf.crs != "EPSG:4326":
            self.geodf = self.geodf.to_crs("EPSG:4326")

        # self.geodf.to_file("dissolved_gadm.gpkg", driver="GPKG")

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

    def generate_heatmap(self, points, s, b):
        heatmap, x_edges, y_edges = np.histogram2d(points[:, 0], points[:, 1], bins=b)  # doesn't histo across world's edge
        heatmap = gaussian_filter(heatmap, sigma=s)

        # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        return heatmap.T

    def plot_sampling(self, load_points=False, n_points=100000, smoothing=0, bins=300, normalize_heatmap=True, show_map=True, show_points=False):
        points_path = os.path.join(self.save_path, f"{n_points}_world_points.npy")
        if not load_points:  # Why haven't I looked into Hammersley sampling?
            n_lats = round(np.sqrt(n_points * 2 / 3))
            n_lngs = round(n_points / n_lats)

            lats = np.linspace(-90, 90, n_lats)
            lngs = np.linspace(-180, 180, n_lngs)
            points = [Point(lng, lat) for lat in lats for lng in lngs]

            used_points = self.simulate_sampling(tuple(points), self.geodf.boundary)
            # boundary.plot()
            # boundary_points = gpd.GeoDataFrame(geometry=self.get_points(boundary), crs=boundary.crs)
        
            np.save(points_path, used_points)
        else:
            used_points = np.load(points_path)

        if show_map:
            ax = self.geodf.plot(alpha=0.2)

            bounds = self.geodf.total_bounds  # NOT [-180, 180, -90, 90] (for some reason doesn't go up to north pole)
            extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
        else:
            ax = plt.gca()  # could just plot imshow first...
            extent = [-180, 180, -90, 90]

        heat_map = self.generate_heatmap(used_points, smoothing, bins)
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
            ax.scatter(used_points[:, 0], used_points[:, 1], color="r", s=1, zorder=4)

        plt.axis("off")

        sampling_path = os.path.join(self.save_path, f"sampling_{n_points}np_{smoothing}smo_{bins}b_{normalize_heatmap}no_{show_map}shm_{show_points}shp.png")
        plt.savefig(sampling_path, dpi=1000, bbox_inches="tight")

        plt.close()
