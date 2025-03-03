import time

from train_components.callbacks.callback import Callback

class TimerCallback(Callback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_train_begin(self, **kwargs):
        self.train_start_time = time.time()

    def on_train_end(self, **kwargs):
        train_end_time = time.time()
        train_total_time = train_end_time - self.train_start_time
        self.train_start_time = None

        print(f"Training took {train_total_time:.2f} seconds")

    def on_epoch_begin(self, epoch, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs, **kwargs):
        epoch_end_time = time.time()
        epoch_total_time = epoch_end_time - self.epoch_start_time
        self.epoch_start_time = None

        print(f"Epoch {epoch} took {epoch_total_time:.2f} seconds")
