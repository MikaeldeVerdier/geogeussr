import os
import numpy as np
import random
import cv2
import tensorflow as tf
from tensorflow.data import Dataset

from shared_components.tokenizer import Tokenizer
from shared_components.files import load_annotations

class DatasetHandler:
    def __init__(self, dataset_path, split, batch_size, regions):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.regions = regions

        self.tokenizer = Tokenizer(regions)

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

    def encode_location(self, location):  # could do this in init to avoid repeating (not that expensive though)
        description = self.generate_description(location)
        tokenized_description = self.tokenizer.encode_texts([description])[0]

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

    def decode_predictions_standard(self, logits_per_image, tokenized_labels):
        best_match_idx = np.argmax(logits_per_image, axis=-1)
        best_match_label = np.array(tokenized_labels)[..., best_match_idx]

        exp_sims = np.exp(logits_per_image)
        norm_sims = exp_sims / np.sum(exp_sims, axis=-1)
        best_match_confidence = norm_sims[..., best_match_idx]

        return best_match_label, best_match_confidence

    def decode_predictions_com(self, logits_per_image, tokenized_labels):  # weird for these two to be different in args and rets
        decoded_texts = []
        for batch_sim in logits_per_image:
            norm_batch_sim = (batch_sim + 1) / 2  # needed? neg sims can't be allowed?

            total_weighted_lats = 0
            total_weighted_lngs = 0
            total_weight = 0
            for similarity, label in zip(norm_batch_sim, tokenized_labels):
                total_weighted_lats += label[1] * similarity
                total_weighted_lngs += label[2] * similarity
                total_weight += similarity

            avg_latitude = total_weighted_lats / total_weight
            avg_longitude = total_weighted_lngs / total_weight

            encoded_text = [-1, avg_latitude, avg_longitude]
            decoded_texts.append(self.tokenizer.decode_texts([encoded_text])[0])

        return decoded_texts
