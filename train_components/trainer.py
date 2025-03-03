import os

import train_components.train_config as train_cfg
from shared_components.dataset_handler import DatasetHandler
from train_components.callbacks.timer_callback import TimerCallback
from train_components.callbacks.checkpoint_callback import ModelCheckpointWithHistoryCallback
from train_components.callbacks.callback_list import CallbackList

class Trainer:
    def __init__(self, processor_method="Geo"):  # should this be in config too?
        train_batch_size = round(train_cfg.computation_batch_size * (1 - train_cfg.validation_split))
        val_batch_size = train_cfg.computation_batch_size - train_batch_size

        self.train_dataset_handler = DatasetHandler(train_cfg.dataset_path, train_cfg.image_size, 1 - train_cfg.validation_split, train_batch_size, processor_method=processor_method)
        self.val_dataset_handler = DatasetHandler(train_cfg.dataset_path, train_cfg.image_size, -train_cfg.validation_split, val_batch_size, processor_method=processor_method)

        # self.log_path = os.path.join(train.SAVE_PATH, "training_log.json")

    def create_timer_callback(self):
        timer_callback = TimerCallback()

        return timer_callback

    def create_checkpoint_callback(self, load, save_interval, name):
        history_filepath = os.path.join(train_cfg.save_dir, name, f"{name}_training_log.json")  # f"{train.SAVE_PATH}/{name}/training_log.json"
        checkpoint_filepath = os.path.join(train_cfg.save_dir, name, f"{name}" + "_{epoch}")
        model_checkpoint_callback = ModelCheckpointWithHistoryCallback(load, history_filepath, model_filepath=checkpoint_filepath, save_interval=save_interval)

        return model_checkpoint_callback

    def create_callbacks(self, load, save_interval, name):
        timer_callback = self.create_timer_callback()
        model_checkpoint_callback = self.create_checkpoint_callback(load, save_interval, name)
        callback_list = CallbackList([timer_callback, model_checkpoint_callback])

        return callback_list

    def train(self, model, load=False):
        # optimizer = self.build_optimizer(**train_cfg.optimizer_config)

        save_interval = int(train_cfg.iteration_amount * train_cfg.save_ratio)
        callbacks = self.create_callbacks(load, save_interval, train_cfg.name)

        start_iteration = 0  # callback.get_epoch()
        end_iteration = start_iteration + train_cfg.iteration_amount

        train_dataset = self.train_dataset_handler.create_dataset(train_cfg.used_regions, use_augmentation=train_cfg.use_augmentation)
        # validation_dataset = self.val_dataset_handler.create_dataset(train_cfg.used_regions, use_augmentation=False)  # no augmentation

        print(f"Training {train_cfg.name} for {train_cfg.iteration_amount} iterations")
        if train_cfg.effective_batch_size is None:
            # model.clip_compile(optimizer=optimizer, loss=ContrastiveLoss())
            # model.clip_fit(
            #     train_dataset,
            #     epochs=end_iteration,
            #     callbacks=[callback],
            #     validation_data=validation_dataset,
            #     initial_epoch=start_iteration,
            #     validation_steps=1,
            #     steps_per_epoch=1
            # )
            pass
        else:
            model.prepare_training(**train_cfg.optimizer_config)
            model.custom_fit(
                train_dataset,
                initial_epoch=start_iteration,
                final_epoch=end_iteration,
                callbacks=callbacks,
                effective_batch_size=train_cfg.effective_batch_size
            )
