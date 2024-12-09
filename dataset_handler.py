import os
import numpy as np
import random
import cv2
import tensorflow as tf
from tensorflow.data import Dataset

from files import load_annotations
from countries import *

class DatasetHandler:
    def __init__(self, dataset_path, split, batch_size):
        self.dataset_path = dataset_path
        self.batch_size = batch_size

        loaded_annotations = load_annotations(dataset_path)
        share = int(len(loaded_annotations) * split)

        self.annotations = loaded_annotations[:share] if split >= 0 else loaded_annotations[share:]

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

    def encode_location(self, location):
        encoded_coords = [location["lat"], location["lng"]]

        return encoded_coords

    def create_generator(self, input_shape, preprocess_function):
        while True:
            chosen_annotations = random.sample(self.annotations, min(self.batch_size, len(self.annotations)))

            x_batch = []
            y_batch = []
            for annotation in chosen_annotations:
                x = self.encode_image(annotation["image_name"], input_shape, preprocess_function)
                x_batch.append(x)

                y = self.encode_location(annotation["location"])
                # y_1_batch.append(y_1)
                y_batch.append(y)

            yield (np.array(x_batch), np.array(y_batch))

    def create_dataset(self, image_size, num_classes, preprocess_function):
        used_batch_size = min(self.batch_size, len(self.annotations))
        if used_batch_size == 0:
            return None

        generator = lambda: self.create_generator(image_size, preprocess_function)  # why does this need to be lambda-wrapped (wrapped at all)?
        dataset = Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(used_batch_size,) + image_size, dtype=tf.float32),
                tf.TensorSpec(shape=(used_batch_size, num_classes), dtype=tf.float32)
            )
        )

        return dataset
