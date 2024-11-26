import os
import json
from copy import copy
from keras.callbacks import ModelCheckpoint

class ModelCheckpointWithHistory(ModelCheckpoint):
    def __init__(self, load_initial, history_filepath, **kwargs):
        super().__init__(**kwargs)

        self.load = load_initial
        self.history_filepath = history_filepath
        self.history = {}

        self.setup_save_folder()

    def setup_save_folder(self):
        save_folders = self.history_filepath.replace("\\", "/").split("/")[:-1]
        for i in range(1, len(save_folders) + 1):  # hate range(len(x)) but yeah
            folder_to_check = "/".join(save_folders[:i])

            if not os.path.exists(folder_to_check):
                os.mkdir(folder_to_check)  # why doesn't os.mkdir just create all folders?

    def append_to_history(self, old_logs, logs, is_test=False):
        appended_logs = {
            f"val_{key}"
            if is_test else key:
            old_logs.get(f"val_{key}" if is_test else key, []) + (value if isinstance(value, list) else [value])  # ugly

            for key, value in logs.items()
        }
        new_logs = old_logs | appended_logs

        return new_logs

    def on_train_batch_end(self, batch, logs=None):
        self.history = self.append_to_history(self.history, logs)

        copy_self = copy(self)  # need to copy because calling _should_save_on_batch has a persistent impact
        if copy_self._should_save_on_batch(batch):
            if self.load and os.path.exists(self.history_filepath):
                with open(self.history_filepath, "r") as f:
                    old_metrics = json.load(f)
            else:
                old_metrics = {}

            used_metrics = self.append_to_history(old_metrics, self.history)  # validation metrics are 1 behind when saving...
            with open(self.history_filepath, "w") as f:
                json.dump(used_metrics, f)

            self.load = True
            self.history = {}  # I load them again when it's time to save next

        super().on_train_batch_end(batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        self.history = self.append_to_history(self.history, logs, is_test=True)

        super().on_test_batch_end(batch, logs)
