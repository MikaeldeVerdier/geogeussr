import os 

MODEL_NAME = "model.keras"
DATASET_PATH = "datasets/dataset"
AMOUNT_ITERATIONS = 10
BATCH_SIZE = 8
VALIDATION_SPLIT = 0.2

SAVE_PATH = "save_folder"
SAVE_RATIO = 1

MODEL_PATH = os.path.join(SAVE_PATH, MODEL_NAME)  # don't like computation in config...
