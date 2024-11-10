from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import plot_model

import configs.training_config as train
import configs.adam_config as adam
from trainer import Trainer
from models.full_model import FullModel
# from models.haversine_loss import HaversineLoss
from models.losses.root_mean_squared_error import RootMeanSquareError
from visualizer.visualize_gmm import visualize_gmm

if __name__ == "__main__":
    # model = load_model(train.MODEL_PATH, custom_objects={"HaversineLoss": HaversineLoss})

    trainer = Trainer(train.DATASET_PATH, train.BATCH_SIZE, train.SHAPEFILE_PATH, train.VALIDATION_SPLIT)
    # visualize_gmm(trainer.dataset_handler.gm, trainer.dataset_handler.coords, save_path="clusters.png", shapefile_path="dataset_generator/gadm_410.gpkg")

    model = FullModel()
    model.build((None, model.input_shape[0], model.input_shape[1], model.input_shape[2]))
    optimizer = Adam(learning_rate=adam.LEARNING_RATE, beta_1=adam.BETA_1, beta_2=adam.BETA_2)
    model.compile(optimizer=optimizer, loss=["categorical_crossentropy", RootMeanSquareError()])
    model.summary()

    trainer.train(model, train.AMOUNT_ITERATIONS)

    model.save(train.MODEL_PATH)
