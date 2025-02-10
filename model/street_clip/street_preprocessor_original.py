import os
import numpy as np
import cv2
from transformers import CLIPProcessor

from model.preprocessor import Preprocessor

class StreetPreprocessorOriginal(Preprocessor):
    def __init__(self, dataset_path, regions, image_size=(336, 336, 3), **kwargs):
        super().__init__(regions, **kwargs)

        self.dataset_path = dataset_path

        self.clip_processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

        transposed_image_size = (image_size[2], image_size[0], image_size[1])
        self.output_shapes = {
            "input_ids": ((None,), int),
            "attention_mask": ((None,), int),
            "pixel_values": (transposed_image_size, float)
        }

    def __call__(self, chosen_annotations, image_shape, use_augmentation=False, rets=[], passed_processed_images=None, passed_images=None, passed_prompts=None, **kwargs):  # could unify with other clip preprocessors
        x_batch = {"input_ids": [], "attention_mask": [], "pixel_values": []}
        y_batch = []

        images = [] if passed_processed_images is None and passed_images is None else None
        locations = [] if passed_prompts is None else passed_prompts
        for annotation in chosen_annotations:
            if passed_processed_images is None and passed_images is None:
                image = self.get_image(annotation["image_name"], image_shape)
                images.append(image)

            if passed_prompts is None:
                location = self.get_location(annotation["location"])
                locations.append(location)

            y = self.generate_description(annotation["location"])
            y_batch.append(y)

        if passed_images is not None:
            images = passed_images

        if passed_prompts is not None:
            locations = self.encode_texts(locations)

        if self.data_augmentor is not None and use_augmentation:
            images = self.data_augmentor(images)

        x_batch = self.process(images, locations)

        if passed_processed_images is not None:
            x_batch["pixel_values"] = passed_processed_images

        if not len(rets):
            return x_batch, np.array(y_batch)

        ret_data = []
        if "raw_images" in rets:
            ret_data.append(images)
        if "raw_locations" in rets:
            ret_data = (locations)

        return x_batch, np.array(y_batch), ret_data  # y_true not used, but is just GT description

    def process(self, image_input, text_input):
        processed = self.clip_processor(text=text_input, images=image_input, return_tensors="np", padding=True)
        processed_data = processed.data

        return processed_data

    def get_image(self, image_name, input_shape):
        image_path = os.path.join(self.dataset_path, image_name)

        img = cv2.imread(image_path)
        img = cv2.resize(img, input_shape[:-1])
        img = img[..., ::-1]

        return img

    def get_location(self, location):  # could do this in init to avoid repeating (not that expensive though)
        description = self.generate_description(location)
        encoded_description = self.encode_texts([description])[0]

        return encoded_description

    def encode_texts(self, texts):
        encoded_texts = []
        for text in texts:
            encoded_text = text.split(", latitude ")[0] + "."

            encoded_texts.append(encoded_text)

        return encoded_texts

    """
    def decode_texts(self, encoded_texts):
        texts = []
        for encoded_text in encoded_texts:
            region = self.inverse_region_translations[encoded_text]
            region_index = self.regions.index(region)
            origin = self.origins[region_index]

            texts.append(f"{region}, latitude {origin[1]}, longitude {origin[0]}")

        return np.array(texts)
    """

    def get_refinement_prompts(self, best_prompt, refinement_amount):
        return self.get_refinement_prompts_city(best_prompt, refinement_amount)

    def find_image(self, inputs):
        return inputs["pixel_values"]
