import os
from tf_keras.optimizers import Adam, SGD
from tf_keras.optimizers.schedules import ExponentialDecay

import train_components.train_config as train_cfg
from model.constrastive_loss import ContrastiveLoss
from shared_components.dataset_handler import DatasetHandler
from train_components.callbacks import ModelCheckpointWithHistory

class Trainer:
    def __init__(self, processor_method="Geo"):  # should this be in config too?
        train_batch_size = round(train_cfg.batch_size * (1 - train_cfg.validation_split))
        val_batch_size = train_cfg.batch_size - train_batch_size

        processor_kwargs = {
            "iamge_size": train_cfg.image_size,
            "max_len": train_cfg.max_len
        }

        self.train_dataset_handler = DatasetHandler(train_cfg.dataset_path, 1 - train_cfg.validation_split, train_batch_size, train_cfg.regions, processor_method=processor_method, processor_kwargs=processor_kwargs)
        self.val_dataset_handler = DatasetHandler(train_cfg.dataset_path, -train_cfg.validation_split, val_batch_size, train_cfg.regions, processor_method=processor_method, processor_kwargs=processor_kwargs)

        # self.log_path = os.path.join(train.SAVE_PATH, "training_log.json")

    def build_optimizer(self, initial_lr, decay_steps, decay_factor, beta_1, beta_2, weight_decay):
        schedule = ExponentialDecay(initial_lr, decay_steps, decay_factor, staircase=True)
        # optimizer = Adam(learning_rate=schedule, beta_1=beta_1, beta_2=beta_2, weight_decay=weight_decay)
        optimizer = SGD(learning_rate=schedule, momentum=beta_1, weight_decay=weight_decay)

        return optimizer

    def create_checkpoint_callback(self, load, save_interval, name):
        history_filepath = os.path.join(train_cfg.save_dir, name, f"{name}_training_log.json")  # f"{train.SAVE_PATH}/{name}/training_log.json"
        checkpoint_filepath = os.path.join(train_cfg.save_dir, name, f"{name}" + "_{epoch}")
        model_checkpoint_callback = ModelCheckpointWithHistory(load, history_filepath, model_filepath=checkpoint_filepath, save_interval=save_interval)

        return model_checkpoint_callback

    def train(self, model, load=False):
        optimizer = self.build_optimizer(train_cfg.initial_learning_rate, train_cfg.decay_steps, train_cfg.decay_factor, train_cfg.beta_1, train_cfg.beta_2, train_cfg.weight_decay)
        model.compile(optimizer=optimizer, loss=ContrastiveLoss())

        save_interval = int(train_cfg.iteration_amount * train_cfg.save_ratio)
        callback = self.create_checkpoint_callback(load, save_interval, train_cfg.name)

        start_iteration = callback.get_epoch()
        end_iteration = start_iteration + train_cfg.iteration_amount

        train_dataset = self.train_dataset_handler.create_dataset(train_cfg.image_size, train_cfg.used_regions)
        validation_dataset = self.val_dataset_handler.create_dataset(train_cfg.image_size, train_cfg.used_regions)

        print(f"Training {train_cfg.name} for {train_cfg.iteration_amount} iterations")
        model.fit(
            train_dataset,
            epochs=end_iteration,
            callbacks=[callback],
            validation_data=validation_dataset,
            initial_epoch=start_iteration,
            validation_steps=1,
            steps_per_epoch=1
        )
