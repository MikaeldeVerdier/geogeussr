import os
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

    def prepare_model(self, model):  # Could save this in annotations and then just load it instead of having to do this double...
        for annotation in self.annotations:
            point = Point(annotation["location"]["lng"], annotation["location"]["lat"])
            country = self.geodf[self.geodf.contains(point)]

            if not len(country.index.values):
                continue

            country_name = country.index.values[0]
            country_index = COUNTRIES.index(country_name)

            if model.specialized_regressors[country_index] is None:
                model.add_regressor(country_index)

        model.build((None, model.input_shape[0], model.input_shape[1], model.input_shape[2]))

    def encode_image(self, image_name, input_shape, preprocess_function):
        image_path = os.path.join(self.dataset_path, image_name)

        img = cv2.imread(image_path)
        img = cv2.resize(img, input_shape[:-1])
        preprocessed_image = preprocess_function(img)

        return preprocessed_image

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

        return one_hot_country, encoded_coords

    def generate_batch(self, input_shape, preprocess_function, split=1):
        while True:
            chosen_annotations = random.sample(self.annotations, int(round(self.batch_size * split)))

            x_batch = []
            y_1_batch = []
            y_2_batch = []
            for annotation in chosen_annotations:
                x = self.encode_image(annotation["image_name"], input_shape, preprocess_function)
                x_batch.append(x)

                y_1, y_2 = self.encode_coords(annotation["location"]["lat"], annotation["location"]["lng"])
                y_1_batch.append(y_1)
                y_2_batch.append(y_2)

            yield np.array(x_batch), (np.array(y_1_batch), np.array(y_2_batch))

    def decode_predictions(self, class_probs, regressed_values):  # doesn't really fit here but this is where shapefile is loaded so
        coords = []
        for batch_probs, batch_vals in zip(class_probs, regressed_values):
            country_name = COUNTRIES[np.argmax(batch_probs, axis=-1)]
            country = self.geodf[self.geodf.index == country_name]

            origin = country.to_crs("EPSG:3857").geometry.centroid.to_crs("EPSG:4326")._values[0]

            local_x = batch_vals[0] * 1000
            local_y = batch_vals[1] * 1000

            proj = pyproj.Proj(proj="aeqd", lat_0=origin.y, lon_0=origin.x)  # could store these from encoding
            lng, lat = proj(local_x, local_y, inverse=True)

            coords.append([lat, lng])

        np_coords = np.array(coords)

        return np_coords
