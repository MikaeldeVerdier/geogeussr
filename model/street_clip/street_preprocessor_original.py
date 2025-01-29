import os
import numpy as np
import cv2
from transformers import CLIPProcessor

class StreetPreprocessorOriginal:
    def __init__(self, dataset_path, regions, region_translations=None, origins=None, image_size=(336, 336, 3), **kwargs):
        self.dataset_path = dataset_path
        self.regions = regions
        self.region_translations = region_translations
        self.origins = origins

        self.inverse_region_translations = {v: k for k, v in region_translations.items()}

        self.clip_processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

        transposed_image_size = (image_size[2], image_size[0], image_size[1])
        self.output_shapes = {
            "input_ids": ((None,), int),
            "attention_mask": ((None,), int),
            "pixel_values": (transposed_image_size, float)
        }

    def __call__(self, chosen_annotations, image_shape, passed_images=None, passed_prompts=None, **kwargs):  # could obtimize by onnly calculating what needs to be returned
        x_batch = {"input_ids": [], "attention_mask": [], "pixel_values": []}
        y_batch = []

        images = [] if passed_images is None else passed_images
        locations = [] if passed_prompts is None else passed_prompts
        for annotation in chosen_annotations:
            if passed_images is None:
                image = self.get_image(annotation["image_name"], image_shape)
                images.append(image)

            if passed_prompts is None:
                location = self.get_location(annotation["location"])
                locations.append(location)

            y = self.generate_description(annotation["location"])
            y_batch.append(y)

        if passed_prompts is not None:
            locations = self.encode_texts(locations)

        x_batch = self.process(images, locations)

        return x_batch, np.array(y_batch)  # y_true not used, but is just GT description

    def process(self, image_input, text_input):
        processed = self.clip_processor(text=text_input, images=image_input, return_tensors="tf", padding=True, do_rescale=False)
        processed_data = processed.data

        return processed_data

    def get_image(self, image_name, input_shape):
        image_path = os.path.join(self.dataset_path, image_name)

        img = cv2.imread(image_path)
        img = cv2.resize(img, input_shape[:-1])
        scaled_img = img / 255.0

        return scaled_img

    def generate_description(self, location):
        return f"{location['country']}, latitude {float(location['lat']):.3f}, longitude {float(location['lng']):.3f}"

    def get_location(self, location):  # could do this in init to avoid repeating (not that expensive though)
        description = self.generate_description(location)
        encoded_description = self.encode_texts([description])[0]

        return encoded_description

    def encode_texts(self, texts):
        encoded_texts = []
        for text in texts:
            components = text.split(", ")
            country = self.region_translations[components[0]]

            encoded_texts.append(country)

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

    def get_components(self, texts):  # format is so inconsistent throughout this class...
        # regions = []
        components = []
        for text in texts:
            text_comps = text.split(", ")
            region = text_comps[0]
            latitude = float(text_comps[1].split(" ")[1])
            longitude = float(text_comps[2].split(" ")[1])

            # regions.append(region)
            components.append([region, latitude, longitude])

        return components
