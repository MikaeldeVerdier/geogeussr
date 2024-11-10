import tensorflow as tf
from keras.models import Model
from keras.applications.vgg16 import preprocess_input

import configs.classifier_configs.cnn_config as cls_cnn_cfg
import configs.classifier_configs.vit_config as cls_vit_cfg
import configs.regressor_configs.cnn_config as reg_cnn_cfg
import configs.regressor_configs.vit_config as reg_vit_cfg
from models.archictectures.cnn_model import ConvolutionalNeuralNetwork
from models.archictectures.vit_model import VisionTransformer

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
        predicted_classes = tf.cast(tf.argmax(class_probs, axis=-1), tf.int32)  # for some reason switch_case requires this

        if not tf.is_symbolic_tensor(predicted_classes):  # TODO: this is so shit
            regressed_values = [
                self.specialized_regressors[idx]
                if self.specialized_regressors[idx] is not None else
                tf.zeros((reg_cnn_cfg.NUM_CLASSES,), dtype=tf.float32)

                for idx in predicted_classes
            ]
        else:
            regressed_values = tf.zeros((tf.shape(inputs)[0], reg_cnn_cfg.NUM_CLASSES), dtype=tf.float32)

        return class_probs, regressed_values

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
