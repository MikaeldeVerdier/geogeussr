import os
import numpy as np
import cv2
from transformers import CLIPProcessor

from model.preprocessor import Preprocessor

class ClipPreprocessor(Preprocessor):
    def __init__(self, dataset_path, image_size=(336, 336, 3), **kwargs):
        super().__init__(**kwargs)

        self.dataset_path = dataset_path
        self.image_size = image_size

        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

        transposed_image_size = (image_size[2], image_size[0], image_size[1])
        self.output_shapes = {  # doing a lot here just to not import tensorflow even though i've improted transformers. should just define tensorSpec here...
            "input_ids": ((None,), int),
            "attention_mask": ((None,), int),
            "pixel_values": (transposed_image_size, float)
        }

    def __call__(self, chosen_annotations, use_augmentation=False, rets=[], passed_processed_images=None, passed_images=None, passed_prompts=None, **kwargs):  # could obtimize by onnly calculating what needs to be returned
        x_batch = {"input_ids": [], "attention_mask": [], "pixel_values": []}
        y_batch = []

        images = [] if passed_processed_images is None and passed_images is None else None
        locations = [] if passed_prompts is None else passed_prompts
        for annotation in chosen_annotations:
            if passed_processed_images is None and passed_images is None:
                image = self.get_image(annotation["image_name"])
                images.append(image)

            if passed_prompts is None:
                location = self.get_location(annotation["location"])
                locations.append(location)

            y = self.get_basic_descriptions(**annotation["location"]["coding"], use_all=False)[0]
            y_batch.append(y)

        if passed_images is not None:
            images = passed_images

        if passed_prompts is not None:
            locations = self.encode_texts(locations)  # not needed, just here for continuity with other preprocessors

        if self.data_augmentor is not None and use_augmentation:
            images = self.data_augmentor(images)

        x_batch = self.process(images, locations)

        if passed_processed_images is not None:
            x_batch["pixel_values"] = passed_processed_images 

        if not len(rets):
            return x_batch, np.array(y_batch)

        ret_data = {}
        if "raw_images" in rets:
            ret_data["raw_images"] = images
        if "raw_locations" in rets:
            ret_data["raw_locations"] = locations
        if "gt_components" in rets:
            locs = []
            for annotation in chosen_annotations:
                loc = annotation["location"]
                coding = list(loc["coding"].values())
                lat_lng = [loc["lat"], loc["lng"]]

                locs.append([coding, lat_lng])

            ret_data["gt_components"] = locs

        return x_batch, np.array(y_batch), ret_data

    def process(self, image_input, text_input):
        processed = self.clip_processor(text=list(text_input), images=image_input, return_tensors="np", padding=True)
        processed_data = processed.data
        # processed_data["input_ids"] = np.pad(processed_data["input_ids"], ((0, 0), (0, 100 - processed_data["input_ids"].shape[1])))
        # processed_data["attention_mask"] = np.pad(processed_data["attention_mask"], ((0, 0), (0, 100 - processed_data["attention_mask"].shape[1])))

        return processed_data

    def get_image(self, image_name):
        image_path = os.path.join(self.dataset_path, image_name)

        img = cv2.imread(image_path)
        img = cv2.resize(img, self.image_size[:-1])
        img = img[..., ::-1]

        return img

    def get_location(self, location):  # could do this in init to avoid repeating (not that expensive though)
        description = self.generate_description(location)

        return description

    def encode_texts(self, texts):
        return np.array(texts)

    """
    def decode_texts(self, texts):
        return np.array(texts)
    """

    def get_refinement_prompts(self, best_prompt, refinement_amount):
        return self.get_refinement_prompts_coords(best_prompt, refinement_amount)

    def find_image(self, inputs):
        return inputs["pixel_values"]
