import os
import numpy as np
import random
import joblib
import cv2
import pyproj
import tensorflow as tf
from tensorflow.data import Dataset

from files import load_annotations
from configs.tool_configs.cluster_config import NUM_CLUSTERS

class DatasetHandler:
    def __init__(self, dataset_path, gm_path, split, batch_size):
        self.dataset_path = dataset_path
        self.batch_size = batch_size

        self.gm = joblib.load(gm_path)

        loaded_annotations = load_annotations(dataset_path)
        share = int(len(loaded_annotations) * split)

        self.annotations = loaded_annotations[:share] if split >= 0 else loaded_annotations[share:]
        self.unique_regions, self.annotation_counts = np.unique([self.gm.predict([[annotation["location"]["lat"], annotation["location"]["lng"]]])[0] for annotation in self.annotations], return_counts=True)

        # self.geodf = gpd.read_file(shapefile_path)
        # self.geodf = self.geodf.dissolve(by="GID_0")
        # if self.geodf.crs != "EPSG:4326":  # used for accurate centroid later
        #     self.geodf = self.geodf.to_crs("EPSG:4326")

        # self.generate_clusters(num_clusters)

    def encode_image(self, image_name, input_shape, preprocess_function):
        image_path = os.path.join(self.dataset_path, image_name)

        img = cv2.imread(image_path)
        img = cv2.resize(img, input_shape[:-1])
        preprocessed_image = preprocess_function(img[None])[0]

        return preprocessed_image

    def encode_location(self, location, index):
        region_index = self.gm.predict([[location["lat"], location["lng"]]])[0]
        one_hot_region = np.eye(NUM_CLUSTERS)[region_index]  # Really should check output_shape for the classifier (num_classes) # one_hot_region = np.zeros(len(COUNTRIES)); one_hot_region[COUNTRIES.index(region_name)] = 1

        if index == 0:
            return one_hot_region

        origin = self.gm.means_[region_index]
        # origin = region.to_crs("EPSG:3857").geometry.centroid.to_crs("EPSG:4326").iloc[0]  # don't like but I think it's fine because it's just one entry
        proj = pyproj.Proj(proj="aeqd", lat_0=origin[0], lon_0=origin[1])  # Azimuthal equidistant projection for accurate (x, y) coordinates
        local_x, local_y = proj(location["lng"], location["lat"])  # DECODE COORDS IS JUST proj(local_x, local_y, inverse=True)

        encoded_coords = np.array([local_x / 1000, local_y / 1000])  # in km now  # to decode: * 1000

        if index == 1:
            return encoded_coords

        return one_hot_region, encoded_coords

    def get_region_annotations(self, region_names):
        if region_names is not None:
            region_annotations = [
                annotation
                for annotation in self.annotations
                if self.gm.predict([[annotation["location"]["lat"], annotation["location"]["lng"]]])[0] in region_names
            ]
        else:
            region_annotations = self.annotations

        return region_annotations

    def create_generator(self, input_shape, preprocess_function, region_names, y_index):
        while True:
            region_annotations = self.get_region_annotations(region_names)
            chosen_annotations = random.sample(region_annotations, min(self.batch_size, len(region_annotations)))

            x_batch = []
            y_batch = []
            for annotation in chosen_annotations:
                x = self.encode_image(annotation["image_name"], input_shape, preprocess_function)
                x_batch.append(x)

                if y_index != 0 and y_index != 1:
                    pred_region = self.gm.predict([[annotation["location"]["lat"], annotation["location"]["lng"]]])[0]
                    y_batch.append(([annotation["location"]["lat"], annotation["location"]["lng"]], pred_region))

                    continue

                y = self.encode_location(annotation["location"], y_index)
                # y_1_batch.append(y_1)
                y_batch.append(y)

            if y_index != 0 and y_index != 1:
                outputs = list(zip(*y_batch))
                np_return = (np.array(x_batch), (np.array(outputs[0]), np.array(outputs[1])))
            else:
                np_return = (np.array(x_batch), np.array(y_batch))

            yield np_return

    def create_dataset(self, input_shape, num_classes, image_size, preprocess_function, region_names, y_index):
        region_annotations = self.get_region_annotations(region_names)  # unecessarily calculated independently twice
        used_batch_size = min(self.batch_size, len(region_annotations))
        if used_batch_size == 0:
            return None
        
        # return self.create_generator(image_size, preprocess_function, region_name, y_index)

        generator = lambda: self.create_generator(image_size, preprocess_function, region_names, y_index)  # why does this need to be lambda-wrapped (wrapped at all)?
        dataset = Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(used_batch_size,) + input_shape, dtype=tf.float32),
                tf.TensorSpec(shape=(used_batch_size, num_classes), dtype=tf.float32)
            )
        )

        return dataset

    def decode_predictions(self, class_probs, regressed_values, ret_region=False, ret_local_coords=False):
        coords = []
        countries = []
        local_coords = []
        for batch_probs, batch_vals in zip(class_probs, regressed_values):
            region_index = np.argmax(batch_probs, axis=-1)

            origin = self.gm.means_[region_index]

            local_x = batch_vals[0] * 1000
            local_y = batch_vals[1] * 1000

            proj = pyproj.Proj(proj="aeqd", lat_0=origin[1], lon_0=origin[0])  # could store these from encoding
            lng, lat = proj(local_x, local_y, inverse=True)

            coords.append([lat, lng])
            if ret_region:
                region_conf = batch_probs[region_index]
                countries.append([region_index], region_conf)
            if ret_local_coords:
                local_coords.append([local_x, local_y])

        if not ret_region or ret_local_coords:
            return np.array(coords)

        ret_vals = [np.array(coords)]
        if ret_region:
            ret_vals.append(countries)
        if ret_local_coords:
            ret_vals.append(local_coords)

        return ret_vals
