import os
from keras.optimizers import Adam, SGD
from keras.optimizers.schedules import ExponentialDecay
from keras import mixed_precision

import train_components.train_config as train_cfg
from model.contrastive_loss import ContrastiveLoss
from shared_components.dataset_handler import DatasetHandler
from train_components.callbacks import ModelCheckpointWithHistory

class Trainer:
    def __init__(self, processor_method="Geo"):  # should this be in config too?
        train_batch_size = round(train_cfg.computation_batch_size * (1 - train_cfg.validation_split))
        val_batch_size = train_cfg.computation_batch_size - train_batch_size

        self.train_dataset_handler = DatasetHandler(train_cfg.dataset_path, train_cfg.image_size, 1 - train_cfg.validation_split, train_batch_size, processor_method=processor_method)
        self.val_dataset_handler = DatasetHandler(train_cfg.dataset_path, train_cfg.image_size, -train_cfg.validation_split, val_batch_size, processor_method=processor_method)

        if train_cfg.use_mixed_precision:
            mixed_precision.set_global_policy("mixed_float16")

        # self.log_path = os.path.join(train.SAVE_PATH, "training_log.json")

    def build_optimizer(self, initial_lr, decay_steps, decay_factor, beta_1, beta_2, weight_decay, **kwargs):
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
        optimizer = self.build_optimizer(**train_cfg.optimizer_config)

        save_interval = int(train_cfg.iteration_amount * train_cfg.save_ratio)
        callback = self.create_checkpoint_callback(load, save_interval, train_cfg.name)

        start_iteration = callback.get_epoch()
        end_iteration = start_iteration + train_cfg.iteration_amount

        train_dataset = self.train_dataset_handler.create_dataset(train_cfg.used_regions, use_augmentation=train_cfg.use_augmentation)
        validation_dataset = self.val_dataset_handler.create_dataset(train_cfg.used_regions, use_augmentation=False)  # no augmentation

        print(f"Training {train_cfg.name} for {train_cfg.iteration_amount} iterations")
        if train_cfg.effective_batch_size is None:
            model.clip_compile(optimizer=optimizer, loss=ContrastiveLoss())
            model.clip_fit(
                train_dataset,
                epochs=end_iteration,
                callbacks=[callback],
                validation_data=validation_dataset,
                initial_epoch=start_iteration,
                validation_steps=1,
                steps_per_epoch=1
            )
        else:
            model.compile(optimizer=optimizer, loss=ContrastiveLoss())
            model.custom_fit(
                train_dataset,
                initial_epoch=start_iteration,
                final_epoch=end_iteration,
                callbacks=[callback],
                effective_batch_size=train_cfg.effective_batch_size
            )
