import numpy as np

from dataset_handler import DatasetHandler

class Trainer:
    def __init__(self, dataset_path, batch_size, shapefile_path, validation_split):
        self.validation_split = validation_split

        self.dataset_handler = DatasetHandler(dataset_path, batch_size, shapefile_path)

    """
    def conditional_add_regressor(self, model, encoded_data):
        country_index = encoded_data[0].tolist().index(1)  # np.where(encoded_data[0] == 1)[0]
        if model.specialized_regressors[country_index] is None:
            model.add_regressor(country_index)
    """

    def train(self, model, iteration_amount):
        for _ in range(iteration_amount):
            batch = self.dataset_handler.generate_batch(model.preprocess_func, model.input_shape)
            x, y = zip(*batch)

            y_countries, y_coords = zip(*y)

            model.fit(np.array(x), [np.array(y_countries), np.array(y_coords)], validation_split=self.validation_split)
