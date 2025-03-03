class Callback:
    def __init__(self):
        pass

    def on_train_begin(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

    def on_epoch_begin(self, epoch, **kwargs):
        pass

    def on_epoch_end(self, epoch, logs, **kwargs):
        pass

    def on_train_batch_begin(self, step, **kwargs):
        pass

    def on_train_batch_end(self, step, **kwargs):
        pass
