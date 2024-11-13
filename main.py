from keras.models import load_model

import configs.training_config as train
import configs.adam_config as adam
from trainer import Trainer
from models.full_model import FullModel
# from models.haversine_loss import HaversineLoss
from models.losses.root_mean_squared_error import RootMeanSquareError
from visualizer.visualize_gmm import visualize_gmm

if __name__ == "__main__":
    # model = load_model(train.MODEL_PATH, custom_objects={"RootMeanSquareError": RootMeanSquareError})

    trainer = Trainer(train.DATASET_PATH, train.BATCH_SIZE, train.SHAPEFILE_PATH, train.VALIDATION_SPLIT)
    # visualize_gmm(trainer.dataset_handler.gm, trainer.dataset_handler.coords, save_path="clusters.png", shapefile_path="dataset_generator/gadm_410.gpkg")

    model = FullModel()
    trainer.dataset_handler.prepare_model(model)

    optimizer = trainer.build_optimizer(adam.INITIAL_LEARNING_RATE, adam.DECAY_STEPS, adam.DECAY_FACTOR, adam.BETA_1, adam.BETA_2)
    model.compile(optimizer=optimizer, loss=["categorical_crossentropy", RootMeanSquareError()])
    # model.summary()

    trainer.train(model, train.AMOUNT_ITERATIONS, train.SAVE_RATIO)

    model.save(train.MODEL_PATH)
