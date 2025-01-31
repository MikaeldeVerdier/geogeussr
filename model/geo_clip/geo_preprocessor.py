import os
import numpy as np
import cv2

class GeoPreprocessor:
    def __init__(self, dataset_path, regions, image_size=(336, 336, 3), max_len=1000, **kwargs):
        self.dataset_path = dataset_path
        self.regions = regions

        self.output_shapes = (
            (image_size, float),
            ((max_len,), int)  # pretty sure this one can't be None
        )

    def __call__(self, chosen_annotations, image_shape, passed_images=None, passed_prompts=None, **kwargs):  # a bit weird to have from annotations as the format but
        x1_batch = [] if passed_images is None else passed_images  # will do it for any falsy value...
        x2_batch = [] if passed_prompts is None else passed_prompts
        y_batch = []
        for annotation in chosen_annotations:
            if passed_images is None:
                x1 = self.encode_image(annotation["image_name"], image_shape)
                x1_batch.append(x1)

            if passed_prompts is None:
                x2 = self.encode_location(annotation["location"])
                x2_batch.append(x2)

            y = self.generate_description(annotation["location"])
            y_batch.append(y)

        if passed_prompts is not None:
            x2_batch = self.encode_texts(x2_batch)

        return (np.array(x1_batch), np.array(x2_batch)), np.array(y_batch)  # y_true not used, but is just GT description

    def encode_image(self, image_name, input_shape):
        image_path = os.path.join(self.dataset_path, image_name)

        img = cv2.imread(image_path)
        img = cv2.resize(img, input_shape[:-1])

        return img / 255.0

    def generate_description(self, location):
        return f"{location['country']}, latitude {location['lat']}, longitude {location['lng']}"

    def encode_location(self, location):  # could do this in init to avoid repeating (not that expensive though)
        description = self.generate_description(location)
        tokenized_description = self.encode_texts([description])[0]

        return tokenized_description

    def encode_texts(self, texts):
        encoded_texts = []
        for text in texts:
            components = text.split(", ")
            country = components[0]
            latitude = float(components[1].split(" ")[1])
            longitude = float(components[2].split(" ")[1])

            country_idx = self.regions.index(country) + 3600
            encoded_lat = np.round(latitude + 90 * 10, 1)  # in the range [0, 1800]
            encoded_lng = np.round(longitude + 180 * 10, 1)  # in the range [0, 3600]

            encoded_texts.append([country_idx, encoded_lat, encoded_lng])

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
