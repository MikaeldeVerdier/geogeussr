import tensorflow as tf
from keras.models import Model
from keras.applications.vgg16 import preprocess_input

import configs.classifier_configs.cnn_config as cls_cnn_cfg
import configs.classifier_configs.vit_config as cls_vit_cfg
import configs.regressor_configs.cnn_config as reg_cnn_cfg
import configs.regressor_configs.vit_config as reg_vit_cfg
from models.cnn_model import ConvolutionalNeuralNetwork
from models.vit_model import VisionTransformer

class FullModel(Model):
    def __init__(self):
        super(FullModel, self).__init__()

        # self.preprocess_function = lambda x: x / 255
        self.preprocess_func = preprocess_input

        self.input_shape = cls_cnn_cfg.IMAGE_SIZE

        # self.classifier = VisionTransformer(cls_vit_cfg.NUM_PATCHES, cls_vit_cfg.PATCH_SIZE, cls_vit_cfg.D_MODEL, cls_vit_cfg.NUM_LAYERS, cls_vit_cfg.NUM_HEADS, cls_vit_cfg.MLP_DIM, cls_vit_cfg.NUM_CLASSES)
        self.classifier = ConvolutionalNeuralNetwork(
            cls_cnn_cfg.IMAGE_SIZE,
            cls_cnn_cfg.UNFROZEN_BASE_LAYERS,
            cls_cnn_cfg.NUM_LAYERS,
            cls_cnn_cfg.DENSE_LAYERS,
            cls_cnn_cfg.NUM_CLASSES,
            cls_cnn_cfg.FINAL_ACTIVATION,
            cls_cnn_cfg.KERNEL_INITIALIZER,
            cls_cnn_cfg.L2_REG
        )

        self.specialized_regressors = [
            None
            # ConvolutionalNeuralNetwork(cls_cnn_cfg.IMAGE_SIZE, reg_cnn_cfg.UNFROZEN_BASE_LAYERS, reg_cnn_cfg.NUM_LAYERS, reg_cnn_cfg.DENSE_LAYERS, reg_cnn_cfg.NUM_CLASSES, reg_cnn_cfg.FINAL_ACTIVATION, reg_cnn_cfg.KERNEL_INITIALIZER, reg_cnn_cfg.L2_REG)
            for _ in range(cls_cnn_cfg.NUM_CLASSES)  # self.classifier.output.shape[1]

            # VisionTransformer(reg_vit_cfg.NUM_PATCHES, reg_vit_cfg.PATCH_SIZE, reg_vit_cfg.D_MODEL, reg_vit_cfg.NUM_LAYERS, reg_vit_cfg.NUM_HEADS, reg_vit_cfg.MLP_DIM, reg_vit_cfg.NUM_CLASSES)
            # for _ in range(cls_vit_cfg.NUM_CLASSES)  # self.classifier.output.shape[1]
        ]

    def call(self, inputs):
        class_probs = self.classifier(inputs)
        predicted_class = tf.argmax(class_probs, axis=-1)

        # regressed_value = tf.switch_case(predicted_class, branch_fns=[
        #     lambda: specialized_regressor(inputs)
        #     for specialized_regressor in self.specialized_regressors
        # ])
        regressor = self.specialized_regressors[predicted_class]

        if regressor is None:
            return class_probs, tf.constant([0, 0])

        regressed_value = regressor(inputs)

        return class_probs, regressed_value

    def add_regressor(self, index):
        regressor = ConvolutionalNeuralNetwork(
            cls_cnn_cfg.IMAGE_SIZE,
            reg_cnn_cfg.UNFROZEN_BASE_LAYERS,
            reg_cnn_cfg.NUM_LAYERS,
            reg_cnn_cfg.DENSE_LAYERS,
            reg_cnn_cfg.NUM_CLASSES,
            reg_cnn_cfg.FINAL_ACTIVATION,
            reg_cnn_cfg.KERNEL_INITIALIZER,
            reg_cnn_cfg.L2_REG
        )
        self.specialized_regressors[index] = regressor
