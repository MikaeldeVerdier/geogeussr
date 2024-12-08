import os
import tensorflow as tf
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input

import configs.classifier_configs.cnn_config as cls_cnn_cfg
import configs.regressor_configs.cnn_config as reg_cnn_cfg
from models.subclassed_model import SubclassedModelJSON
from models.archictectures.cnn_model import ConvolutionalNeuralNetwork

from countries import *

class FullModel(SubclassedModelJSON):
    def __init__(self, image_size, num_unfrozen_base_layers, initialize_submodels=False):
        super().__init__(image_size, num_unfrozen_base_layers, initialize_submodels=initialize_submodels)

        self.used_input_shape = image_size
        self.num_unfrozen_base_layers = num_unfrozen_base_layers

        # self.preprocess_function = lambda x: x / 255
        self.preprocess_func = preprocess_input

        base_network = VGG16(include_top=False, weights="imagenet", input_shape=self.used_input_shape)
        self.base_layers = base_network.layers[1:-num_unfrozen_base_layers]  # doesn't support model_cfg.UNFROZEN_BASE_LAYERS == 0 currently...
        for base_layer in self.base_layers:
            base_layer.trainable = False

        # self.classifier = VisionTransformer(cls_vit_cfg.NUM_PATCHES, cls_vit_cfg.PATCH_SIZE, cls_vit_cfg.D_MODEL, cls_vit_cfg.NUM_LAYERS, cls_vit_cfg.NUM_HEADS, cls_vit_cfg.MLP_DIM, cls_vit_cfg.NUM_CLASSES)
        if initialize_submodels:
            self.classifier = ConvolutionalNeuralNetwork(
                image_size,
                num_unfrozen_base_layers,
                cls_cnn_cfg.CONV_LAYERS,
                cls_cnn_cfg.DENSE_LAYERS,
                cls_cnn_cfg.NUM_CLASSES,
                cls_cnn_cfg.FINAL_ACTIVATION,
                cls_cnn_cfg.KERNEL_INITIALIZER,
                cls_cnn_cfg.L2_REG
            )
        else:
            self.classifier = None

        # if initialize_submodels:
        self.specialized_regressors = [
            None
            # ConvolutionalNeuralNetwork(cls_cnn_cfg.IMAGE_SIZE, reg_cnn_cfg.UNFROZEN_BASE_LAYERS, reg_cnn_cfg.NUM_LAYERS, reg_cnn_cfg.DENSE_LAYERS, reg_cnn_cfg.NUM_CLASSES, reg_cnn_cfg.FINAL_ACTIVATION, reg_cnn_cfg.KERNEL_INITIALIZER, reg_cnn_cfg.L2_REG)
            for _ in range(cls_cnn_cfg.NUM_CLASSES)  # self.classifier.output.shape[1]

            # VisionTransformer(reg_vit_cfg.NUM_PATCHES, reg_vit_cfg.PATCH_SIZE, reg_vit_cfg.D_MODEL, reg_vit_cfg.NUM_LAYERS, reg_vit_cfg.NUM_HEADS, reg_vit_cfg.MLP_DIM, reg_vit_cfg.NUM_CLASSES)
            # for _ in range(cls_vit_cfg.NUM_CLASSES)  # self.classifier.output.shape[1]
        ]

    def save(self, save_path):  # could rename to save_incomplete but then save will still be available and save_complete should probably not be allowed because then two different load_incomplete methods would be needed
        overrides = {"initialize_submodels": False}

        super().save(save_path, **overrides)

    @classmethod
    def load_incomplete(cls, save_path):  # could allow this to take overrides too
        overrides = {"initialize_submodels": False}

        return cls.load(save_path, **overrides)

    @classmethod
    def load_complete(cls, save_path):
        model = cls.load_incomplete(save_path)

        classifier_path = os.path.join(save_path, "classifier", "classifier.keras")
        classifier = ConvolutionalNeuralNetwork.load(classifier_path)  # COMMENT FOR COMPATIBILITY
        # classifier = ConvolutionalNeuralNetwork.load(classifier_path, custom_objects={"ConvolutionalNeuralNetwork": ConvolutionalNeuralNetwork})  # UNCOMMENT FOR COMPATIBILITY
        model.classifier = classifier

        for i, country_name in enumerate(COUNTRIES):
            regressor_path = os.path.join(save_path, country_name, f"{country_name}.keras")
            if not os.path.exists(regressor_path):
                continue

            regressor = ConvolutionalNeuralNetwork.load(regressor_path)  # COMMENT FOR COMPATIBILITY
            # regressor = ConvolutionalNeuralNetwork.load(regressor_path, custom_objects={"ConvolutionalNeuralNetwork": ConvolutionalNeuralNetwork})  # UNCOMMENT FOR COMPATIBILITY
            model.specialized_regressors[i] = regressor

        return model

    def base_process(self, inputs):  # is pretty much just first part of call
        x = preprocess_input(inputs)

        for layer in self.base_layers:
            x = layer(x)

        return x.numpy()

    @tf.function
    def call(self, inputs):
        x = inputs
        for layer in self.base_layers:
            x = layer(x)
        base_outputs = x

        class_probs = self.classifier(base_outputs)
        predicted_classes = tf.cast(tf.argmax(class_probs, axis=-1), tf.int32)  # for some reason switch_case requires this

        """
        if not tf.is_symbolic_tensor(predicted_classes):  # this is so shit
            regressed_values = [
                self.specialized_regressors[idx](inputs)
                if self.specialized_regressors[idx] is not None else
                tf.zeros((reg_cnn_cfg.NUM_CLASSES,), dtype=tf.float32)

                for idx in predicted_classes
            ]

            tf.print("NOT SYMBOLIC")
        else:
            regressed_values = tf.zeros((tf.shape(inputs)[0], reg_cnn_cfg.NUM_CLASSES), dtype=tf.float32)
        """

        # regressors = tf.gather(self.specialized_regressors, predicted_classes)
        # regressed_values = tf.vectorized_map(regressors, inputs)
        # # regressed_values = tf.switch_case(predicted_classes, lamb_regressors)

        regressed_values = tf.zeros((tf.shape(base_outputs)[0], reg_cnn_cfg.NUM_CLASSES), dtype=tf.float32)
        for idx, regressor in enumerate(self.specialized_regressors):
            if regressor is None:
                continue

            used_mask = predicted_classes == idx
            if tf.reduce_any(used_mask):
                masked_inputs = tf.boolean_mask(base_outputs, used_mask)
                masked_outputs = regressor(masked_inputs)

                output_mask_indices = tf.where(used_mask)
                regressed_values = tf.tensor_scatter_nd_update(regressed_values, output_mask_indices, masked_outputs)

        return class_probs, regressed_values

    # def get_config(self):
    #     config = super().get_config()
    #     config.update({
    #         "classifier": self.classifier,
    #         "specialized_regressors": self.specialized_regressors
    #     })

    #     return config

    # @classmethod
    # def from_config(cls, config):
    #     classifier = config.pop("classifier")
    #     specialized_regressors = config.pop("specialized_regressors")

    #     return cls(classifier=classifier, specialized_regressors=specialized_regressors, **config)

    def create_classifier(self):
        regressor = ConvolutionalNeuralNetwork(
            self.used_input_shape,
            self.num_unfrozen_base_layers,
            cls_cnn_cfg.CONV_LAYERS,
            cls_cnn_cfg.DENSE_LAYERS,
            cls_cnn_cfg.NUM_CLASSES,
            cls_cnn_cfg.FINAL_ACTIVATION,
            cls_cnn_cfg.KERNEL_INITIALIZER,
            cls_cnn_cfg.L2_REG
        )

        return regressor

    def create_regressor(self):
        regressor = ConvolutionalNeuralNetwork(
            self.used_input_shape,
            self.num_unfrozen_base_layers,
            reg_cnn_cfg.CONV_LAYERS,
            reg_cnn_cfg.DENSE_LAYERS,
            reg_cnn_cfg.NUM_CLASSES,
            reg_cnn_cfg.FINAL_ACTIVATION,
            reg_cnn_cfg.KERNEL_INITIALIZER,
            reg_cnn_cfg.L2_REG
        )

        return regressor

    # def add_regressor(self, index):
    #     print(f"ADDING REGRESSOR FOR: {COUNTRIES[index]}")

    #     regressor = self.create_regressor()
    #     self.specialized_regressors[index] = regressor

    @staticmethod
    def load_submodel(save_path, name, *args, **kwargs):
        path = os.path.join(save_path, name, f"{name}.keras")
        if os.path.exists(path):
            return ConvolutionalNeuralNetwork.load(path, *args, **kwargs)  # COMMENT FOR COMPATIBILITY
            # return ConvolutionalNeuralNetwork.load(path, custom_objects={"ConvolutionalNeuralNetwork": ConvolutionalNeuralNetwork}, *args, **kwargs)  # UNCOMMENT FOR COMPATIBILITY

        return None
