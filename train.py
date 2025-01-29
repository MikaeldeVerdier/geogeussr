from model.geo_clip.geo_clip_model import GeoCLIP
from model.street_clip.street_clip_model import StreetCLIP
from train_components.trainer import Trainer

if __name__ == "__main__":
    geo_clip = GeoCLIP()
    trainer = Trainer()

    trainer.train(geo_clip)

    # street_clip = StreetCLIP()
    # trainer = Trainer(processor_method="Street")

    # trainer.train(street_clip)
