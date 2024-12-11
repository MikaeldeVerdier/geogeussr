import os
import json
from keras.callbacks import Callback

class ModelCheckpointWithHistory(Callback):
    def __init__(self, load_initial, history_filepath, model_filepath, save_interval, **kwargs):
        super().__init__(**kwargs)

        self.load = load_initial
        self.history_filepath = history_filepath
        self.model_filepath = model_filepath
        self.save_interval = save_interval

        self.history = self.load_metrics() if load_initial else {}
        self.num_unsaved_epochs = 0

        self.setup_save_folder()

    def get_epoch(self):
        vals = self.history.values()
        epoch = len(list(vals)[0]) if len(vals) else 0

        return epoch

    def setup_save_folder(self):
        save_folders = self.history_filepath.replace("\\", "/").split("/")[:-1]
        for i in range(1, len(save_folders) + 1):  # hate range(len(x)) but yeah
            folder_to_check = "/".join(save_folders[:i])

            if not os.path.exists(folder_to_check):
                os.mkdir(folder_to_check)  # why doesn't os.mkdir just create all folders?

    def append_to_history(self, old_logs, logs, is_test=False):
        appended_logs = {
            key: old_logs.get(key, []) + [value]
            for key, value in logs.items()
        }
        new_logs = old_logs | appended_logs

        return new_logs

    def load_metrics(self):
        with open(self.history_filepath, "r") as f:
            old_metrics = json.load(f)

        return old_metrics

    def save_metrics(self):
        with open(self.history_filepath, "w") as f:
            json.dump(self.history, f)

    def on_epoch_end(self, epoch, logs=None):
        self.history = self.append_to_history(self.history, logs)
        self.num_unsaved_epochs += 1

        if self.num_unsaved_epochs >= self.save_interval:  # could just use (epoch + 1) % self.save_iterval since epoch is correct now
            self.save_metrics()
            self.model.save(self.model_filepath)

            self.num_unsaved_epochs = 0
