import numpy as np
import random
import tensorflow as tf
from tensorflow.data import Dataset

from data.data_handler import DataHandler
from train_components.data_augmentor import DataAugmentor
# from model.geo_clip.geo_preprocessor import GeoPreprocessor
from model.street_clip.street_preprocessor import StreetPreprocessor
from model.street_clip.street_preprocessor_original import StreetPreprocessorOriginal
from model.clip_clip.clip_preprocessor import ClipPreprocessor
from model.clip_clip.clip_preprocessor_original import ClipPreprocessorOriginal
from shared_components.files import load_annotations

class DatasetHandler:
    def __init__(self, dataset_path, image_size, split, batch_size, data_augmentor_kwargs={}, processor_method="Geo", gadm_path=None, city_path=None):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.gadm_path = gadm_path
        self.city_path = city_path

        self.data_augmentor = DataAugmentor(image_size, **data_augmentor_kwargs)

        if processor_method == "Street":
            self.preprocessor = StreetPreprocessor(dataset_path, image_size, data_augmentor=self.data_augmentor)
        elif processor_method == "StreetOG":
            self.preprocessor = StreetPreprocessorOriginal(dataset_path, image_size, data_augmentor=self.data_augmentor)
        elif processor_method == "Clip":
            self.preprocessor = ClipPreprocessor(dataset_path, image_size, data_augmentor=self.data_augmentor)
        elif processor_method == "ClipOG":
            self.preprocessor = ClipPreprocessorOriginal(dataset_path, image_size, data_augmentor=self.data_augmentor)
        else:
            self.preprocessor = GeoPreprocessor(dataset_path, image_size, data_augmentor=self.data_augmentor)

        loaded_annotations = load_annotations(dataset_path)
        share = int(len(loaded_annotations) * split)

        self.annotations = loaded_annotations[:share] if split >= 0 else loaded_annotations[share:]

        self.data_handler = None

    def get_region_annotations(self, region_names):
        if region_names is None:
            return self.annotations

        region_annotations = [
            annotation
            for annotation in self.annotations
            if annotation["location"]["coding"]["country"] in region_names
        ]

        return region_annotations

    def create_generator(self, region_names, shuffle=True, use_augmentation=False, rets=[], processor_kwargs={}):
        i = 0

        while True:
            region_annotations = self.get_region_annotations(region_names)
            if shuffle:
                chosen_annotations = random.sample(region_annotations, min(self.batch_size, len(region_annotations)))
            else:
                chosen_annotations = [region_annotations[j % len(region_annotations)] for j in range(i, i + self.batch_size)]

            i += 1

            yield self.preprocessor(chosen_annotations, use_augmentation=use_augmentation, rets=rets, **processor_kwargs)

    def create_tensor_spec(self, shape, dtype):
        dtype_map = {
            int: tf.int32,
            float: tf.float32,
            bool: tf.bool,
            str: tf.string,
            bytes: tf.string
        }

        return tf.TensorSpec(shape=(None,) + shape, dtype=dtype_map[dtype])

    def get_output_signature(self, output_shapes):
        if isinstance(output_shapes, list):
            return [self.create_tensor_spec(v, d) for v, d in output_shapes]
        elif isinstance(output_shapes, tuple):
            return tuple(self.get_output_signature(list(output_shapes)))
        elif isinstance(output_shapes, dict):
            return {k: self.create_tensor_spec(v, d) for k, (v, d) in output_shapes.items()} 

    def create_dataset(self, region_names, shuffle=False, use_augmentation=False, rets=[], processor_kwargs={}):
        region_annotations = self.get_region_annotations(region_names)  # unecessarily calculated independently twice
        used_batch_size = min(self.batch_size, len(region_annotations))
        if used_batch_size == 0:
            return None
        
        # return self.create_generator(image_size, preprocess_function, region_name, y_index)

        generator = lambda: self.create_generator(region_names, shuffle=shuffle, use_augmentation=use_augmentation, rets=rets, processor_kwargs=processor_kwargs)  # why does this need to be lambda-wrapped (wrapped at all)?
        output_signature = self.get_output_signature(self.preprocessor.output_shapes)
        dataset = Dataset.from_generator(
            generator,
            output_signature=(
                # (
                #     tf.TensorSpec(shape=(used_batch_size,) + image_shape, dtype=tf.float32),
                #     tf.TensorSpec(shape=(used_batch_size, max_tokens), dtype=tf.int32)
                # ),
                output_signature,
                tf.TensorSpec(shape=(used_batch_size,), dtype=tf.string)  # y_true - doesn't matter (needs to have batch_size as first dimension though, (in some versions))
            )
        )

        return dataset

    def softmax(self, distribution):
        exp_distribution = np.exp(distribution - np.max(distribution))
        softmax_distribution = exp_distribution / np.sum(exp_distribution, axis=-1, keepdims=True)

        return softmax_distribution

    def decode_predictions_standard(self, logits_per_image, prompts, prompt_components):
        best_match_idx = np.argmax(logits_per_image, axis=-1)
        best_match_label = np.array(prompts)[best_match_idx]
        best_match_components = [prompt_components[idx] for idx in best_match_idx]  # hate not being able to vectorize

        norm_sims = self.softmax(logits_per_image)
        best_match_confidence = norm_sims[..., best_match_idx]

        return best_match_label, best_match_components, best_match_confidence

    def decode_predictions_com(self, logits_per_image, prompts, prompt_components):  # center-of-mass approach, uses confidences for all prompts instead of just the best one        
        if self.data_handler is None:  # could be invoked at an earlier point but this works nicely
            self.data_handler = DataHandler(self.city_path, self.gadm_path)

        refinement_level = len(prompt_components[0])  # could be passed as an argument

        lat_lngs = np.array([components[1] for components in prompt_components])

        decoded_texts = []
        decoded_components = []
        for batch_sim in logits_per_image:
            norm_sims = self.softmax(batch_sim)
            avg_point = self.data_handler.spherically_averaged_centroid(lat_lngs, weights=norm_sims)

            coding = self.data_handler.annotate_point(avg_point.y, avg_point.x, force_point=True)
            components = [list(coding.values())[:refinement_level], [avg_point.y, avg_point.x]]
            decoded_components.append(components)  # depending on order like this is bad, perhaps all codings should be dicts

            descriptions = self.preprocessor.get_basic_descriptions(**coding, use_all=False)
            decoded_texts.append(descriptions[0])

        return decoded_texts, decoded_components
