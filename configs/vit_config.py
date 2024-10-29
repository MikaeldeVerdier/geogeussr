IMAGE_SIZE = 640  # Input image dimensions
PATCH_SIZE = 16  # Size of each patch
NUM_CLASSES = 2  # Latitude and Longitude (for regression)
D_MODEL = 64  # Embedding dimension for patches
NUM_HEADS = 4  # Number of attention heads in the transformer
NUM_LAYERS = 8  # Number of transformer layers
MLP_DIM = 128  # Hidden dimension size in MLP head
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # Amount of patches
