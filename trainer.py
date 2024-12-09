import os
# from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay

import configs.model_configs.adam_config as adam
from models.losses.root_mean_squared_error import RootMeanSquareError
from dataset_handler import DatasetHandler
from callbacks import ModelCheckpointWithHistory
from countries import *

class Trainer:
    def __init__(self, dataset_path, validation_split, batch_size, save_path):
        self.save_path = save_path

        train_batch_size = round(batch_size * (1 - validation_split))
        val_batch_size = batch_size - train_batch_size

        self.train_dataset_handler = DatasetHandler(dataset_path, 1 - validation_split, train_batch_size)
        self.val_dataset_handler = DatasetHandler(dataset_path, -validation_split, val_batch_size)

        # self.log_path = os.path.join(self.save_path, "training_log.json")

    def build_optimizer(self, initial_lr, decay_steps, decay_factor, beta_1, beta_2):
        schedule = ExponentialDecay(initial_lr, decay_steps, decay_factor, staircase=True)
        optimizer = Adam(learning_rate=schedule, beta_1=beta_1, beta_2=beta_2)

        return optimizer

    def create_checkpoint_callback(self, load, save_freq, name):
        history_filepath = os.path.join(self.save_path, f"{name}_training_log.json")  # f"{self.save_path}/{name}/training_log.json"
        checkpoint_filepath = os.path.join(self.save_path, f"{name}.keras")  # "{epoch}" ?
        model_checkpoint_callback = ModelCheckpointWithHistory(
            load_initial=load,
            history_filepath=history_filepath,
            filepath=checkpoint_filepath,
            verbose=1,
            save_freq=save_freq
        )

        return model_checkpoint_callback

    def train(self, model, load, iteration_amount, save_ratio, name="regressor"):
        loss = RootMeanSquareError()
        class_weights = None  # lat more important than lng?
        start_iteration = 0

        optimizer = self.build_optimizer(adam.INITIAL_LEARNING_RATE, adam.DECAY_STEPS, adam.DECAY_FACTOR, adam.BETA_1, adam.BETA_2)
        model.compile(optimizer=optimizer, loss=loss)

        checkpoint_callback = self.create_checkpoint_callback(load, int(iteration_amount * save_ratio), name)

        train_dataset = self.train_dataset_handler.create_dataset(model.image_size, model.num_classes, model.preprocess_func)
        validation_dataset = self.val_dataset_handler.create_dataset(model.image_size, model.num_classes, model.preprocess_func)

        print(f"Training {name} for {iteration_amount} iterations")
        model.fit(
            train_dataset,
            epochs=iteration_amount,
            class_weight=class_weights,
            callbacks=[checkpoint_callback],
            validation_data=validation_dataset,
            initial_epoch=start_iteration,
            validation_steps=1,
            steps_per_epoch=1
        )
