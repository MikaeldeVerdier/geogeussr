import json
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay

import configs.training_config as train
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

    def build_optimizer(self, initial_lr, decay_steps, decay_factor, beta_1, beta_2):
        schedule = ExponentialDecay(initial_lr, decay_steps, decay_factor, staircase=True)
        optimizer = Adam(learning_rate=schedule, beta_1=beta_1, beta_2=beta_2)

        return optimizer

    def load_metrics(self):
        with open(f"{train.SAVE_PATH}/metrics.json", "r") as f:
            metrics = json.load(f)

        return metrics

    def save_metrics(self, metrics):
        with open(f"{train.SAVE_PATH}/metrics.json", "r") as f:
            json.dump(metrics, f)

    def create_checkpoint_callback(self, save_freq):
        checkpoint_filepath = f"{train.SAVE_PATH}/{train.MODEL_NAME}"
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor="val_loss",
            verbose=1,
            mode="min",
            save_freq=save_freq,
            save_best_only=True
        )

        return model_checkpoint_callback

    def train(self, model, iteration_amount, save_ratio):  # kinda makes this class redundant when using a generator...
        metrics = self.load_metrics()

        checkpoint_callback = self.create_checkpoint_callback(int(iteration_amount * save_ratio))
        history = model.fit(
            self.dataset_handler.generate_batch(model.input_shape, model.preprocess_func),
            epochs=iteration_amount,
            callbacks=[checkpoint_callback],
            steps_per_epoch=1,
            validation_split=self.validation_split
        )
        metrics |= history

        self.save_metrics(metrics)
