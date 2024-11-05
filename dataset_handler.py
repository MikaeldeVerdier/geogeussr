import random
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

from file_decoder import load_annotations

class DatasetHandler:
    def __init__(self, dataset_path, batch_size, num_clusters=10):
        self.dataset_path = dataset_path
        self.batch_size = batch_size

        self.annoations = load_annotations(dataset_path)
        self.generate_clusters(num_clusters)

    def generate_clusters(self, num_clusters):
        self.coords = np.array([[annotation["location"]["lat"], annotation["location"]["lng"]] for annotation in self.annotations])

        self.gm = GaussianMixture(n_components=num_clusters, random_state=0)  # Could generate a bigger sample of annotations by only fetching metadata and saving location and then mixture that
        self.gm = self.gm.fit(self.coords)

    def encode_coords(self, lat, lng):  # Encode with clusters
        return [lat, lng]

    def generate_batch(self, preprocess_function):
        chosen_annotations = random.sample(self.annotations, self.batch_size)

        batch = []
        for annotation in chosen_annotations:
            img = cv2.imread(annotation["image_path"])
            preprocessed_image = preprocess_function(img)

            lat_lng = self.encode_coords(annotation["location"]["lat"], annotation["location"]["lng"])

            batch.append([preprocessed_image, lat_lng])

        return batch
