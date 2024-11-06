import numpy as np

from dataset_handler import DatasetHandler

class Trainer:
    def __init__(self, dataset_path, batch_size, shapefile_path, validation_split):
        self.validation_split = validation_split

        self.dataset_handler = DatasetHandler(dataset_path, batch_size, shapefile_path)

    def train(self, model, iteration_amount):
        for _ in range(iteration_amount):
            batch = self.dataset_handler.generate_batch(model.preprocess_func)
            x, y = zip(*batch)

            model.fit(np.array(x), y, validation_split=self.validation_split)
