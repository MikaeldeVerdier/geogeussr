import tensorflow as tf
tf.config.set_visible_devices([], "GPU")  # because entire model will be model, gpu memory probably won't be enough

import os

import configs.runtime_configs.testing_config as test_cfg
import configs.model_configs.model_config as model_cfg
from evalutator import Evaluator
from models.archictectures.cnn_model import ConvolutionalNeuralNetwork

if __name__ == "__main__":
    model = ConvolutionalNeuralNetwork.load(os.path.join(test_cfg.SAVE_PATH, f"{model_cfg.NAME}.keras"))

    evaluator = Evaluator(test_cfg.DATASET_PATH)
    evaluator.evaluate(model, test_cfg.AMOUNT_ITERATIONS)
