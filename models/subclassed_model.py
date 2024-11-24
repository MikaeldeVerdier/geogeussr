from keras.models import Model, load_model

class SubclassedModel(Model):
    def __init__(self, **kwargs):
        super().__init__()

        self.config = kwargs
    
    def get_functional_model(self):
        inp = self.layers[0].input  # only allows for saving after it's been built, but that should be okay for now.
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