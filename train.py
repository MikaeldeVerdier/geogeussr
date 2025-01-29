from model.geo_clip.geo_clip_model import GeoCLIP
from model.street_clip.street_clip_model import StreetCLIP
from train_components.trainer import Trainer

if __name__ == "__main__":
    # Train a GeoCLIP model from scratch
    geo_clip = GeoCLIP()
    trainer = Trainer()

    trainer.train(geo_clip)

    # # Fine-tune a StreetCLIP model with new prompt format
    # street_clip = StreetCLIP()
    # trainer = Trainer(processor_method="Street")

    # trainer.train(street_clip)

    # # Fine-tune a StreetCLIP model with original prompt format
    # street_clip = StreetCLIP()
    # trainer = Trainer(processor_method="StreetOG")

    # trainer.train(street_clip)

