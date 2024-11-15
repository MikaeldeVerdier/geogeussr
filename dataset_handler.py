import os
import numpy as np
import random
import cv2
import pyproj

from files import load_annotations
from countries import *

class DatasetHandler:
    def __init__(self, dataset_path, batch_size):
        self.dataset_path = dataset_path
        self.batch_size = batch_size

        self.annotations = load_annotations(dataset_path)

        # self.geodf = gpd.read_file(shapefile_path)
        # self.geodf = self.geodf.dissolve(by="GID_0")
        # if self.geodf.crs != "EPSG:4326":  # used for accurate centroid later
        #     self.geodf = self.geodf.to_crs("EPSG:4326")

        # self.generate_clusters(num_clusters)

    """
    def generate_clusters(self, num_clusters):
        self.coords = np.array([[annotation["location"]["lat"], annotation["location"]["lng"]] for annotation in self.annotations])

        self.gm = GaussianMixture(n_components=num_clusters, random_state=0)  # Could generate a bigger sample of annotations by only fetching metadata and saving location and then mixture that
        self.gm = self.gm.fit(self.coords)
    """

    def prepare_model(self, model):  # Could save this in annotations and then just load it instead of having to do this double...
        for annotation in self.annotations:
            country_name = annotation["location"]["country"]
            country_index = COUNTRIES.index(country_name)

            if model.specialized_regressors[country_index] is None:
                model.add_regressor(country_index)

        model.build((None, model.used_input_shape[0], model.used_input_shape[1], model.used_input_shape[2]))

    def encode_image(self, image_name, input_shape, preprocess_function):
        image_path = os.path.join(self.dataset_path, image_name)

        img = cv2.imread(image_path)
        img = cv2.resize(img, input_shape[:-1])
        preprocessed_image = preprocess_function(img)

        return preprocessed_image

    def encode_coords(self, country_name, lat, lng):
        country_index = COUNTRIES.index(country_name)
        one_hot_country = np.eye(len(COUNTRIES))[country_index]  # one_hot_country = np.zeros(len(COUNTRIES)); one_hot_country[COUNTRIES.index(country_name)] = 1

        origin = COUNTRY_ORIGINGS[country_index]  # don't like but I think it's fine because it's just one entry
        # origin = country.to_crs("EPSG:3857").geometry.centroid.to_crs("EPSG:4326").iloc[0]  # don't like but I think it's fine because it's just one entry
        proj = pyproj.Proj(proj="aeqd", lat_0=origin[1], lon_0=origin[0])  # Azimuthal equidistant projection for accurate (x, y) coordinates
        local_x, local_y = proj(lng, lat)  # DECODE COORDS IS JUST proj(local_x, local_y, inverse=True)

        encoded_coords = np.array([local_x / 1000, local_y / 1000])  # in km now  # to decode: * 1000

        return one_hot_country, encoded_coords

    def generate_batch(self, input_shape, preprocess_function, batch_size):
        while True:
            chosen_annotations = random.sample(self.annotations, batch_size)

            x_batch = []
            y_1_batch = []
            y_2_batch = []
            for annotation in chosen_annotations:
                x = self.encode_image(annotation["image_name"], input_shape, preprocess_function)
                x_batch.append(x)

                y_1, y_2 = self.encode_coords(annotation["location"]["country"], annotation["location"]["lat"], annotation["location"]["lng"])
                y_1_batch.append(y_1)
                y_2_batch.append(y_2)

            np_return = (np.array(x_batch), (np.array(y_1_batch), np.array(y_2_batch)))

            yield np_return

    def decode_predictions(self, class_probs, regressed_values):  # doesn't really fit here but this is where shapefile is loaded so
        coords = []
        for batch_probs, batch_vals in zip(class_probs, regressed_values):
            country_index = np.argmax(batch_probs, axis=-1)

            origin = COUNTRY_ORIGINGS[country_index]

            local_x = batch_vals[0] * 1000
            local_y = batch_vals[1] * 1000

            proj = pyproj.Proj(proj="aeqd", lat_0=origin[1], lon_0=origin[0])  # could store these from encoding
            lng, lat = proj(local_x, local_y, inverse=True)

            coords.append([lat, lng])

        np_coords = np.array(coords)

        return np_coords
