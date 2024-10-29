from keras.models import load_model
from keras.optimizers import Adam

import configs.training_config as train
import configs.cnn_config as cnn
import configs.vit_config as vit
import configs.adam_config as adam
from trainer import Trainer
from models.cnn_model import ConvolutionalNeuralNetwork
from models.vit_model import VisionTransformer
from models.haversine_loss import HaversineLoss

if __name__ == "__main__":
    # model = load_model(train.MODEL_PATH, custom_objects={"HaversineLoss": HaversineLoss})

    # model = VisionTransformer(vit.NUM_PATCHES, vit.PATCH_SIZE, vit.D_MODEL, vit.NUM_LAYERS, vit.NUM_HEADS, vit.MLP_DIM, vit.NUM_CLASSES)
    model = ConvolutionalNeuralNetwork(cnn.IMAGE_SIZE, cnn.UNFROZEN_BASE_LAYERS, cnn.NUM_LAYERS, cnn.DENSE_LAYERS, cnn.NUM_CLASSES, cnn.KERNEL_INITIALIZER, cnn.L2_REG)
    model.compile(optimizer=Adam(learning_rate=adam.LEARNING_RATE, beta_1=adam.BETA_1, beta_2=adam.BETA_2), loss=HaversineLoss())

    trainer = Trainer(train.DATASET_PATH, train.BATCH_SIZE, train.VALIDATION_SPLIT)
    trainer.train(model, train.AMOUNT_ITERATIONS)

    model.save(train.MODEL_PATH)
