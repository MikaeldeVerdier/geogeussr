class CallbackList:  # inspired by keras.callbacks.CallbackList
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def on_train_begin(self, **kwargs):
        for callback in self.callbacks:
            callback.on_train_begin(**kwargs)

    def on_train_end(self, **kwargs):
        for callback in self.callbacks:
            callback.on_train_end(**kwargs)

    def on_epoch_begin(self, epoch, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, **kwargs)

    def on_epoch_end(self, epoch, logs, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs, **kwargs)

    def on_train_batch_begin(self, step, **kwargs):
        for callback in self.callbacks:
            callback.on_train_batch_begin(step, **kwargs)

    def on_train_batch_end(self, step, **kwargs):
        for callback in self.callbacks:
            callback.on_train_batch_end(step, **kwargs)
