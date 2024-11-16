import json
from keras.callbacks import ModelCheckpoint

class ModelCheckpointWithHistory(ModelCheckpoint):
    def __init__(self, history_filepath, **kwargs):
        super().__init__(**kwargs)

        self.history_filepath = history_filepath
        self.history = {}

    def on_train_batch_end(self, batch, logs=None):
        if self._should_save_on_batch(batch):
            self.history = {key: self.history.get(key, []) + [value] for key, value in logs.items()}

            with open(self.history_filepath, 'w') as f:
                json.dump(self.history, f)

        super().on_train_batch_end(batch, logs)
