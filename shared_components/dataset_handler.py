import os
import numpy as np
import random
import cv2
import tensorflow as tf
from tensorflow.data import Dataset

from shared_components.files import load_annotations

class DatasetHandler:
    def __init__(self, dataset_path, split, batch_size, regions):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.regions = regions

        loaded_annotations = load_annotations(dataset_path)
        share = int(len(loaded_annotations) * split)

        self.annotations = loaded_annotations[:share] if split >= 0 else loaded_annotations[share:]
        self.unique_regions, self.annotation_counts = np.unique([annotation["location"]["country"] for annotation in self.annotations], return_counts=True)

    def encode_image(self, image_name, input_shape):
        image_path = os.path.join(self.dataset_path, image_name)

        img = cv2.imread(image_path)
        img = cv2.resize(img, input_shape[:-1])

        return img / 255.0

    def generate_description(self, location):
        return f"{location['country']}, latitude {location['lat']}, longitude {location['lng']}"

    def tokenize_description(self, description):  # could just skip this and have this in encode
        components = description.split(", ")
        country = components[0]
        latitude = float(components[1].split(" ")[1])
        longitude = float(components[2].split(" ")[1])

        country_idx = self.regions.index(country)
        lat_norm = (latitude + 90) / 180  # Normalize to [0, 1]
        lon_norm = (longitude + 180) / 360  # Normalize to [0, 1]

        return [country_idx, lat_norm, lon_norm]

    def encode_location(self, location):  # could do this in init to avoid repeating (not that expensive though)
        description = self.generate_description(location)
        tokenized_description = self.tokenize_description(description)

        return tokenized_description

    def get_region_annotations(self, region_names):
        if region_names is not None:
            region_annotations = [
                annotation
                for annotation in self.annotations
                if annotation["location"]["country"] in region_names
            ]
        else:
            region_annotations = self.annotations

        return region_annotations

    def create_generator(self, image_shape, region_names):
        while True:
            region_annotations = self.get_region_annotations(region_names)
            chosen_annotations = random.sample(region_annotations, min(self.batch_size, len(region_annotations)))

            x_batch = []
            y_batch = []
            for annotation in chosen_annotations:
                x = self.encode_image(annotation["image_name"], image_shape)
                x_batch.append(x)

                y = self.encode_location(annotation["location"])
                y_batch.append(y)

            yield (np.array(x_batch), np.array(y_batch)), np.zeros((len(x_batch)))  # y_true not used

    def create_dataset(self, image_shape, max_tokens, region_names):
        region_annotations = self.get_region_annotations(region_names)  # unecessarily calculated independently twice
        used_batch_size = min(self.batch_size, len(region_annotations))
        if used_batch_size == 0:
            return None
        
        # return self.create_generator(image_size, preprocess_function, region_name, y_index)

        generator = lambda: self.create_generator(image_shape, region_names)  # why does this need to be lambda-wrapped (wrapped at all)?
        dataset = Dataset.from_generator(
            generator,
            output_signature=(
                (
                    tf.TensorSpec(shape=(used_batch_size,) + image_shape, dtype=tf.float32),
                    tf.TensorSpec(shape=(used_batch_size, max_tokens), dtype=tf.float32)
                ),
                tf.TensorSpec(shape=(used_batch_size,))  # y_true - doesn't matter (needs to have batch_size as first dimension though, (in some versions))
            )
        )

        return dataset

    """  # TODO: Add this
    def decode_predictions(self, class_probs, regressed_values, ret_region=False, ret_local_coords=False):
        coords = []
        countries = []
        local_coords = []
        for batch_probs, batch_vals in zip(class_probs, regressed_values):
            region_index = np.argmax(batch_probs, axis=-1)

            origin = COUNTRY_ORIGINGS[region_index]
            local_x = batch_vals[0] * 1000
            local_y = batch_vals[1] * 1000

            proj = pyproj.Proj(proj="aeqd", lat_0=origin[1], lon_0=origin[0])  # could store these from encoding
            lng, lat = proj(local_x, local_y, inverse=True)

            coords.append([lat, lng])
            if ret_region:
                region_conf = batch_probs[region_index]
                countries.append([region_index, region_conf])
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
        """
