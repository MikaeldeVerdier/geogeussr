import os
import numpy as np
import random
import cv2
import tensorflow as tf
from tensorflow.data import Dataset

from shared_components.geo_tokenizer import GeoTokenizer
from shared_components.street_tokenizer import StreetTokenizer
from shared_components.files import load_annotations

class DatasetHandler:
    def __init__(self, dataset_path, split, batch_size, regions, region_translations=None, region_origins=None, tokenizer_method="Geo"):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.regions = regions

        if tokenizer_method == "Geo":
            self.tokenizer = GeoTokenizer(regions)
        else:
            self.tokenizer = StreetTokenizer(regions, region_translations, region_origins)

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

            x1_batch = []
            x2_batch = []
            y_batch = []
            for annotation in chosen_annotations:
                x1 = self.encode_image(annotation["image_name"], image_shape)
                x1_batch.append(x1)

                x2 = self.encode_location(annotation["location"])
                x2_batch.append(x2)

                y = self.generate_description(annotation["location"])
                y_batch.append(y)

            yield (np.array(x1_batch), np.array(x2_batch)), np.array(y_batch)  # y_true not used, but is just GT description

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
                    tf.TensorSpec(shape=(used_batch_size, max_tokens), dtype=tf.int32)
                ),
                tf.TensorSpec(shape=(used_batch_size,))  # y_true - doesn't matter (needs to have batch_size as first dimension though, (in some versions))
            )
        )

        return dataset

    def softmax(self, distribution):
        exp_distribution = np.exp(distribution)
        softmax_distribution = exp_distribution / np.sum(exp_distribution, axis=-1)

        return softmax_distribution

    def decode_predictions_standard(self, logits_per_image, prompts):
        best_match_idx = np.argmax(logits_per_image, axis=-1)
        best_match_label = np.array(prompts)[..., best_match_idx]

        norm_sims = self.softmax(logits_per_image)
        best_match_confidence = norm_sims[..., best_match_idx]

        return best_match_label, best_match_confidence

    def decode_predictions_com(self, logits_per_image, prompts):  # weird for these two to be different in args and rets
        decoded_texts = []
        for batch_sim in logits_per_image:
            # norm_batch_sim = (batch_sim + 1) / 2  # normalizes to range [0, 1]. needed? neg sims can't be allowed?
            norm_batch_sim = self.softmax(batch_sim)

            total_weighted_lats = 0
            total_weighted_lngs = 0
            total_weight = 0
            for similarity, prompt in zip(norm_batch_sim, prompts):
                components = self.tokenizer.get_components([prompt])[0]

                total_weighted_lats += components[1] * similarity
                total_weighted_lngs += components[2] * similarity
                total_weight += similarity

            avg_latitude = total_weighted_lats / total_weight
            avg_longitude = total_weighted_lngs / total_weight

            best_prompt = np.array(prompts)[np.argmax(batch_sim)]
            best_prompt_region = self.tokenizer.get_components([best_prompt])[0][0]
            encoded_text = {"country": best_prompt_region, "lat": avg_latitude, "lng": avg_longitude}
            decoded_texts.append(self.generate_description(encoded_text))

        return decoded_texts
