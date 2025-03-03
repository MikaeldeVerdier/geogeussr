import numpy as np
import random

from data.data_handler import DataHandler
from shared_components.torch_dataset import TorchDataset
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

    def create_dataset(self, region_names, shuffle=False, use_augmentation=False, rets=[], processor_kwargs={}):
        if self.batch_size == 0:
            return None

        dataset = TorchDataset(self.annotations, self.preprocessor, region_names=region_names, data_augmentor=self.data_augmentor, use_augmentation=use_augmentation, rets=rets, processor_kwargs=processor_kwargs)
        loader = dataset.get_loader(self.batch_size, shuffle=shuffle)

        return loader

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
