# Model config
name = "GeoCLIP"
image_size = (336, 336, 3)
used_regions = None  # ["Sweden"] to only train on Sweden

# Training config
dataset_path = "data/datasets/dataset10"
use_augmentation = False
use_mixed_precision = False
iteration_amount = 180_000
batch_size = 2
validation_split = 0
save_ratio = 0.05
save_dir = "save_folder"

# Optimizer config
optimizer_config = {
    "initial_lr": 1e-4,
    "decay_steps": 1000,
    "decay_factor": 0.95,
    "beta_1": 0.9,
    "beta_2": 0.98,
    "weight_decay": 0.1
}
