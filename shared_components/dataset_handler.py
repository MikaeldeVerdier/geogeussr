import numpy as np
import random
import tensorflow as tf
import geopandas as gpd
from tensorflow.data import Dataset
from shapely import Point

from model.geo_clip.geo_preprocessor import GeoPreprocessor
from model.street_clip.street_preprocessor import StreetPreprocessor
from model.street_clip.street_preprocessor_original import StreetPreprocessorOriginal
from model.clip_clip.clip_preprocessor import ClipPreprocessor
from model.clip_clip.clip_preprocessor_original import ClipPreprocessorOriginal
from shared_components.files import load_annotations

class DatasetHandler:
    def __init__(self, dataset_path, split, batch_size, regions, processor_method="Geo", processor_kwargs={}, shapefile_path=None):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.regions = regions

        if processor_method == "Street":
            self.preprocessor = StreetPreprocessor(dataset_path, regions, **processor_kwargs)
        elif processor_method == "StreetOG":
            self.preprocessor = StreetPreprocessorOriginal(dataset_path, regions, **processor_kwargs)
        elif processor_method == "Clip":
            self.preprocessor = ClipPreprocessor(dataset_path, regions, **processor_kwargs)
        elif processor_method == "ClipOG":
            self.preprocessor = ClipPreprocessorOriginal(dataset_path, regions, **processor_kwargs)
        else:
            self.preprocessor = GeoPreprocessor(dataset_path, regions, **processor_kwargs)

        loaded_annotations = load_annotations(dataset_path)
        share = int(len(loaded_annotations) * split)

        self.annotations = loaded_annotations[:share] if split >= 0 else loaded_annotations[share:]
        self.unique_regions, self.annotation_counts = np.unique([annotation["location"]["country"] for annotation in self.annotations], return_counts=True)

        self.load_gadm(shapefile_path)

    def load_gadm(self, shapefile_path):
        if shapefile_path is None:
            return

        self.geodf = gpd.read_file(shapefile_path)  # Weird for this to use gadm directly when everything else has to rely on the information in regions variable
        if self.geodf.crs != "EPSG:3857":
            self.geodf = self.geodf.to_crs("EPSG:3857")

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

    def create_generator(self, image_shape, region_names, rets=[], processor_kwargs={}):
        while True:
            region_annotations = self.get_region_annotations(region_names)
            chosen_annotations = random.sample(region_annotations, min(self.batch_size, len(region_annotations)))

            yield self.preprocessor(chosen_annotations, image_shape, rets=rets, **processor_kwargs)

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

    def create_dataset(self, image_shape, region_names, processor_kwargs={}):
        region_annotations = self.get_region_annotations(region_names)  # unecessarily calculated independently twice
        used_batch_size = min(self.batch_size, len(region_annotations))
        if used_batch_size == 0:
            return None
        
        # return self.create_generator(image_size, preprocess_function, region_name, y_index)

        generator = lambda: self.create_generator(image_shape, region_names, processor_kwargs=processor_kwargs)  # why does this need to be lambda-wrapped (wrapped at all)?
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

    def decode_predictions_standard(self, logits_per_image, prompts):
        best_match_idx = np.argmax(logits_per_image, axis=-1)
        best_match_label = np.array(prompts)[..., best_match_idx]

        norm_sims = self.softmax(logits_per_image)
        best_match_confidence = norm_sims[..., best_match_idx]

        return best_match_label, best_match_confidence

    def region_for_point(self, point):  # POINT IN FORMAT [LONGITUDE, LATITUDE] (could reverse for cosistency)
        if self.geodf is not None:
            shp_point_ll = gpd.GeoDataFrame(geometry=[Point(point[0], point[1])], crs="EPSG:4326")
            shp_point = shp_point_ll.to_crs("EPSG:3857").geometry.loc[0]  # don't like using loc
            distances = self.geodf.geometry.distance(shp_point)
            country_name = self.geodf.loc[distances.idxmin()]["GID_0"]  # don't like using loc

            return country_name

        # the following won't always be correct. be weary of using it (not speicying shapefile_path)
        bounding_boxes = np.array([region["bounding_box"] for region in self.regions])
        inside_x = (point[0] >= bounding_boxes[:, 0]) & (point[0] <= bounding_boxes[:, 2])
        inside_y = (point[1] >= bounding_boxes[:, 1]) & (point[1] <= bounding_boxes[:, 3])  # could combine these statements for minor performance imrpovement

        fitting_region_indices = np.where(inside_x & inside_y)[0]  # can be true for multiple regions, choose the one with the lowest area.
        fitting_regions = np.array(self.regions)[fitting_region_indices]
        fitting_region_areas = (bounding_boxes[fitting_region_indices][:, 2] - bounding_boxes[fitting_region_indices][:, 0]) * (bounding_boxes[fitting_region_indices][:, 3] - bounding_boxes[fitting_region_indices][:, 1])
        lowest_area_region_code = fitting_regions[np.argmin(fitting_region_areas)]  # Could do a ratio between the area and the distance to the center of the region instead

        return lowest_area_region_code

    def decode_predictions_com(self, logits_per_image, prompts):  # center-of-mass approach, uses confidences for all prompts instead of just the best one        
        decoded_texts = []
        for batch_sim in logits_per_image:
            # norm_batch_sim = (batch_sim + 1) / 2  # normalizes to range [0, 1]. needed? neg sims can't be allowed?
            norm_batch_sim = self.softmax(batch_sim)

            # total_weighted_lats = 0
            # total_weighted_lngs = 0
            # total_weight = 0  // total_weight is always 1 after softmax
            components = np.array(self.preprocessor.get_components(prompts))
            lats = np.array(components[:, 1], dtype=np.float32)
            lngs = np.array(components[:, 2], dtype=np.float32)  # TODO: This method is flawed. doesn't take looping around the globe into account.
            avg_latitude = np.sum(lats * norm_batch_sim, axis=-1)
            avg_longitude = np.sum(lngs * norm_batch_sim, axis=-1)
            # for similarity, prompt in zip(norm_batch_sim, prompts):
            #     components = self.preprocessor.get_components([prompt])[0]

            #     total_weighted_lats += components[1] * similarity
            #     total_weighted_lngs += components[2] * similarity
            #     # total_weight += similarity

            # avg_latitude = total_weighted_lats  # / total_weight
            # avg_longitude = total_weighted_lngs  # / total_weight

            # best_prompt = np.array(prompts)[np.argmax(batch_sim)]
            # best_prompt_region = self.preprocessor.get_components([best_prompt])[0][0]
            region = self.region_for_point([avg_longitude, avg_latitude])
            encoded_text = {
                "country": region,  # Caution: may not always be in the right region. Could check which country the lat/lng lands in instead but could be expensive
                "lat": avg_latitude,
                "lng": avg_longitude
            }

            decoded_texts.append(self.preprocessor.generate_description(encoded_text))

        return decoded_texts
