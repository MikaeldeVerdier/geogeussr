import tensorflow as tf
tf.config.set_visible_devices([], "GPU")  # because entire model will be model, gpu memory probably won't be enough

import os
from keras.models import load_model

import configs.runtime_configs.testing_config as test_cfg
import configs.model_configs.model_config as model_cfg
from evalutator import Evaluator

if __name__ == "__main__":
    model = load_model(os.path.join(test_cfg.SAVE_PATH, model_cfg.NAME))

    evaluator = Evaluator(test_cfg.DATASET_PATH)
    evaluator.evaluate(model, test_cfg.AMOUNT_ITERATIONS)
