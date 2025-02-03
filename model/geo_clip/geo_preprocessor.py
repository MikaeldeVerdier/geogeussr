import os
import numpy as np
import cv2

from model.preprocessor import Preprocessor

class GeoPreprocessor(Preprocessor):
    def __init__(self, dataset_path, regions, image_size=(336, 336, 3), max_len=1000, **kwargs):
        super().__init__(regions, **kwargs)

        self.dataset_path = dataset_path

        self.output_shapes = (
            (image_size, float),
            ((max_len,), int)  # pretty sure this one can't be None
        )

    def __call__(self, chosen_annotations, image_shape, passed_processed_images=None, passed_prompts=None, **kwargs):  # a bit weird to have from annotations as the format but
        x1_batch = [] if passed_processed_images is None else None  # will do it for any falsy value...
        x2_batch = [] if passed_prompts is None else passed_prompts
        y_batch = []
        for annotation in chosen_annotations:
            if passed_processed_images is None:
                x1 = self.encode_image(annotation["image_name"], image_shape)
                x1_batch.append(x1)

            if passed_prompts is None:
                x2 = self.encode_location(annotation["location"])
                x2_batch.append(x2)

            y = self.generate_description(annotation["location"])
            y_batch.append(y)

        if passed_prompts is not None:
            x2_batch = self.encode_texts(x2_batch)

        if passed_processed_images is not None:
            x1_batch = passed_processed_images

        return (np.array(x1_batch), np.array(x2_batch)), np.array(y_batch)  # y_true not used, but is just GT description

    def encode_image(self, image_name, input_shape):
        image_path = os.path.join(self.dataset_path, image_name)

        img = cv2.imread(image_path)
        img = cv2.resize(img, input_shape[:-1])

        return img / 255.0

    def encode_location(self, location):  # could do this in init to avoid repeating (not that expensive though)
        description = self.generate_description(location)
        tokenized_description = self.encode_texts([description])[0]

        return tokenized_description

    def encode_texts(self, texts):
        componentss = self.get_components(texts)
        encoded_texts = []
        for components in componentss:
            region_idx = self.get_region_index("code", components[0]) + 3601
            encoded_lat = int(np.round((components[1] + 90) * 10))  # in the range [0, 1800]
            encoded_lng = int(np.round((components[2] + 180) * 10))  # in the range [0, 3600]

            encoded_texts.append([region_idx, encoded_lat, encoded_lng])

        return np.array(encoded_texts)

    """
    def decode_texts(self, encoded_texts):
        texts = []
        for encoded_text in encoded_texts:
            country = self.regions[int(encoded_text[0] - 3600)] if encoded_text[0] != -1 else "Unknown"
            latitude = encoded_text[1] / 10
            longitude = encoded_text[1] / 10

            texts.append(f"{country}, latitude {latitude}, longitude {longitude}")

        return np.array(texts)
    """

    def get_refinement_prompts(self, best_prompt, refinement_amount):
        return self.get_refinement_prompts_coords(best_prompt, refinement_amount)

    def find_image(self, inputs):
        return inputs[0]
