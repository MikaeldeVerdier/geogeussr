from countries import *

IMAGE_SIZE = (640, 640, 3)
UNFROZEN_BASE_LAYERS = 20
NUM_LAYERS = 10
DENSE_LAYERS = [1024, 512]
NUM_CLASSES = len(COUNTRIES)
FINAL_ACTIVATION = "softmax"
KERNEL_INITIALIZER = "he_normal"
L2_REG = 1e-5
