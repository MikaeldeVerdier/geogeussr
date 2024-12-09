from countries import *

CONV_LAYERS = [
    (64, (7, 7), (2, 2), "same", "relu"),
    *[(64, (3, 3), (1, 1), "same", "relu") for _ in range(10)],
    (128, (3, 3), (2, 2), "same", "relu"),
    *[(128, (3, 3), (1, 1), "same", "relu") for _ in range(10)],
    (256, (3, 3), (2, 2), "same", "relu"),
    *[(256, (3, 3), (1, 1), "same", "relu") for _ in range(10)],
    (512, (3, 3), (2, 2), "same", "relu"),
    *[(512, (3, 3), (1, 1), "same", "relu") for _ in range(10)],
]
DENSE_LAYERS = [2048, 1024, 1024, 512]
NUM_CLASSES = len(COUNTRIES)
FINAL_ACTIVATION = "softmax"
KERNEL_INITIALIZER = "he_normal"
L2_REG = 0
