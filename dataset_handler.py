import os
import json
import random
import cv2

class DatasetHandler:
    def __init__(self, dataset_path, batch_size):
        self.dataset_path = dataset_path
        self.batch_size = batch_size

        self.load_annotations()

    def load_annotations(self):
        with open(os.path.join(self.dataset_path, "_annotations.json"), "r") as json_file:
            self.annotations = json.load(json_file)

    def generate_batch(self, preprocess_function):
        chosen_annotations = random.sample(self.annotations, self.batch_size)

        batch = []
        for annotation in chosen_annotations:
            img = cv2.imread(annotation["image_path"])
            preprocessed_image = preprocess_function(img)

            lat_lng = [annotation["location"]["lat"], annotation["location"]["lng"]]

            batch.append([preprocessed_image, lat_lng])

        return batch
