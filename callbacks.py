import os
import json
from copy import copy
from keras.callbacks import ModelCheckpoint

class ModelCheckpointWithHistory(ModelCheckpoint):
    def __init__(self, history_filepath, **kwargs):
        super().__init__(**kwargs)

        self.history_filepath = history_filepath
        self.history = {}

        self.setup_save_folder()

    def setup_save_folder(self):
        save_folders = self.history_filepath.replace("\\", "/").split("/")[:-1]
        for i in range(1, len(save_folders) + 1):  # hate range(len(x)) but yeah
            folder_to_check = "/".join(save_folders[:i])

            if not os.path.exists(folder_to_check):
                os.mkdir(folder_to_check)  # why doesn't os.mkdir just create all folders?

    def append_to_history(self, logs):
        self.history |= {
            key: self.history.get(key, []) + [value]
            for key, value in logs.items()
        }

    def on_train_batch_end(self, batch, logs=None):
        copy_self = copy(self)  # need to copy because calling _should_save_on_batch has a persistent impact
        if copy_self._should_save_on_batch(batch):
            self.append_to_history(logs)

            with open(self.history_filepath, 'w') as f:
                json.dump(self.history, f)

        super().on_train_batch_end(batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        self.append_to_history(logs)

        super().on_test_batch_end(batch, logs)
