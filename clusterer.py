import numpy as np
import joblib
from sklearn.mixture import GaussianMixture

from files import load_annotations

class Clusterer:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters

    def cluster(self, dataset):
        self.gm = GaussianMixture(n_components=self.num_clusters)
        self.gm = self.gm.fit(dataset)

    def cluster_location(self, dataset_path):
        annos = load_annotations(dataset_path)
        self.coords = np.array([[anno["location"]["lat"], anno["location"]["lng"]] for anno in annos])

        self.cluster(self.coords)

    def save_gmm(self, save_path):
        joblib.dump(self.gm, save_path)

    def load_gmm(self, load_path):
        self.gm = joblib.load(load_path)
