import os
import json

from keras.models import Model, load_model
# from keras.layers import Input  # UNCOMMENT FOR COMPATIBILITY

class SubclassedModel(Model):
    def __init__(self, **kwargs):
        super().__init__()

        self.config = kwargs

    def get_functional_model(self):  # only allows for saving after it's been built, but that should be okay for now.
        inp = self.layers[0].input  # Input(shape=self.layers[0].input.shape[1:])  # UNCOMMENT FOR COMPATIBILITY
        out = self(inp)
        func_model = Model(inputs=inp, outputs=out)

        return func_model

    def save(self, *args, **kwargs):
        func_model = self.get_functional_model()

        func_model.save(*args, **kwargs)

    @classmethod
    def load(cls, *args, **kwargs):
        func_model = load_model(*args, **kwargs)
        subclassed_model = func_model.layers[-1]

        return subclassed_model

    def get_config(self):
        return self.config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SubclassedModelJSON(Model):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.config = {
            "args": args,
            "kwargs": kwargs
        }

    def save(self, save_path, **overrides):
        used_config = self.config
        used_config["kwargs"] |= overrides

        model_config_path = os.path.join(save_path, "model_config.json")
        with open(model_config_path, "w") as f:
            json.dump(used_config, f)
    
    @classmethod
    def load(cls, save_path, **overrides):
        model_config_path = os.path.join(save_path, "model_config.json")
        with open(model_config_path, "r") as f:
            config = json.load(f)

        args = config["args"]
        kwargs = config["kwargs"]

        kwargs |= overrides

        return cls(*args, **kwargs)
