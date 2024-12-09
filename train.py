import os
from keras.models import load_model

import configs.runtime_configs.training_config as train
import configs.model_configs.model_config as model_cfg
import configs.model_configs.cnn_config as cnn_cfg
from trainer import Trainer
from models.archictectures.cnn_model import ConvolutionalNeuralNetwork

if __name__ == "__main__":
    model_path = os.path.join(train.SAVE_PATH, model_cfg.NAME)
    load = False

    # Create a trainer (always needed)
    trainer = Trainer(train.DATASET_PATH, train.VALIDATION_SPLIT, train.BATCH_SIZE, train.SAVE_PATH)

    # Create a full model (always needed)
    if not load:
        model = ConvolutionalNeuralNetwork(
            model_cfg.IMAGE_SIZE,
            model_cfg.UNFROZEN_BASE_LAYERS,
            cnn_cfg.CONV_LAYERS,
            cnn_cfg.DENSE_LAYERS,
            cnn_cfg.NUM_CLASSES,
            cnn_cfg.FINAL_ACTIVATION,
            cnn_cfg.KERNEL_INITIALIZER,
            cnn_cfg.L2_REG
        )
    else:
        model = load_model(model_path)

    trainer.train(model, load, train.AMOUNT_ITERATIONS, train.SAVE_RATIO, name=model_cfg.NAME)

    model.save(model_path)
