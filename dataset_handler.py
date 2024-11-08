import numpy as np
import random
import cv2
import geopandas as gpd
import pyproj
from shapely.geometry import Point

from files import load_annotations
from countries import *

class DatasetHandler:
    def __init__(self, dataset_path, batch_size, shapefile_path):
        self.dataset_path = dataset_path
        self.batch_size = batch_size

        self.annotations = load_annotations(dataset_path)

        self.geodf = gpd.read_file(shapefile_path)
        self.geodf = self.geodf.dissolve(by="GID_0")
        if self.geodf.crs != "EPSG:4326":  # used for accurate centroid later
            self.geodf = self.geodf.to_crs("EPSG:4326")

        # self.generate_clusters(num_clusters)

    """
    def generate_clusters(self, num_clusters):
        self.coords = np.array([[annotation["location"]["lat"], annotation["location"]["lng"]] for annotation in self.annotations])

        self.gm = GaussianMixture(n_components=num_clusters, random_state=0)  # Could generate a bigger sample of annotations by only fetching metadata and saving location and then mixture that
        self.gm = self.gm.fit(self.coords)
    """

    def encode_coords(self, lat, lng):
        point = Point(lng, lat)
        country = self.geodf[self.geodf.contains(point)]

        country_name = country.index.values[0]
        one_hot_country = np.eye(len(COUNTRIES))[COUNTRIES.index(country_name)]  # one_hot_country = np.zeros(len(COUNTRIES)); one_hot_country[COUNTRIES.index(country_name)] = 1

        origin = country.to_crs("EPSG:3857").geometry.centroid.to_crs("EPSG:4326")._values[0]  # don't like but I think it's fine because it's just one entry
        # origin = country.to_crs("EPSG:3857").geometry.centroid.to_crs("EPSG:4326").iloc[0]  # don't like but I think it's fine because it's just one entry
        proj = pyproj.Proj(proj="aeqd", lat_0=origin.y, lon_0=origin.x)  # Azimuthal equidistant projection for accurate (x, y) coordinates
        local_x, local_y = proj(lng, lat)  # DECODE COORDS IS JUST proj(local_x, local_y, inverse=True)

        encoded_coords = np.array([local_x / 1000, local_y / 1000])  # in km now  # to decode: * 1000

        return [one_hot_country, encoded_coords]

    def generate_batch(self, preprocess_function, input_shape, completion=None):  # this general 'completion' approach is a bit different from the rest of the codebase
        chosen_annotations = random.sample(self.annotations, self.batch_size)

        batch = []
        for annotation in chosen_annotations:
            img = cv2.imread(annotation["image_path"])
            img = cv2.resize(img, input_shape)
            preprocessed_image = preprocess_function(img)

            encoded_data = self.encode_coords(annotation["location"]["lat"], annotation["location"]["lng"])

            batch.append([preprocessed_image, encoded_data])

            if completion is not None:
                completion(encoded_data)

        return batch
