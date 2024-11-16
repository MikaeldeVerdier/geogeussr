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
        self.unique_countries, self.annotation_counts = np.unique([annotation["location"]["country"] for annotation in self.annotations], return_counts=True)

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

    def encode_image(self, image_name, input_shape, preprocess_function):
        image_path = os.path.join(self.dataset_path, image_name)

        img = cv2.imread(image_path)
        img = cv2.resize(img, input_shape[:-1])
        preprocessed_image = preprocess_function(img[None])[0]

        return preprocessed_image

    def encode_location(self, location, index):
        country_index = COUNTRIES.index(location["country"])
        one_hot_country = np.eye(len(COUNTRIES))[country_index]  # Really should check output_shape for the classifier (num_classes) # one_hot_country = np.zeros(len(COUNTRIES)); one_hot_country[COUNTRIES.index(country_name)] = 1

        if index == 0:
            return one_hot_country

        origin = COUNTRY_ORIGINGS[country_index]  # don't like but I think it's fine because it's just one entry
        # origin = country.to_crs("EPSG:3857").geometry.centroid.to_crs("EPSG:4326").iloc[0]  # don't like but I think it's fine because it's just one entry
        proj = pyproj.Proj(proj="aeqd", lat_0=origin[1], lon_0=origin[0])  # Azimuthal equidistant projection for accurate (x, y) coordinates
        local_x, local_y = proj(location["lng"], location["lat"])  # DECODE COORDS IS JUST proj(local_x, local_y, inverse=True)

        encoded_coords = np.array([local_x / 1000, local_y / 1000])  # in km now  # to decode: * 1000

        return encoded_coords

    def generate_batch(self, input_shape, preprocess_function, country_name, y_index, batch_size):
        if country_name is not None:
            country_annotations = [annotation for annotation in self.annotations if annotation["location"]["country"] == country_name]
        else:
            country_annotations = self.annotations

        while True:
            chosen_annotations = random.sample(country_annotations, min(batch_size, len(country_annotations)))

            x_batch = []
            y_batch = []
            for annotation in chosen_annotations:
                x = self.encode_image(annotation["image_name"], input_shape, preprocess_function)
                x_batch.append(x)

                y = self.encode_location(annotation["location"], y_index)
                # y_1_batch.append(y_1)
                y_batch.append(y)

            np_return = (np.array(x_batch), np.array(y_batch))

            yield np_return

    def decode_predictions(self, class_probs, regressed_values):
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
