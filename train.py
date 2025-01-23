from model.model import GeoCLIP
from train_components.trainer import Trainer

if __name__ == "__main__":
    geo_clip = GeoCLIP()
    trainer = Trainer()

    trainer.train(geo_clip)
