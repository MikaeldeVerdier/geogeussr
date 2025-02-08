from shared_components.files import load_json
from model.geo_clip.submodels.configs.vit_config import input_shape
from model.geo_clip.submodels.configs.ttt_config import max_len

# Model config
name = "GeoCLIP"
image_size = input_shape
tokens_len = max_len

# Training config
region_path = "data/country_data.json"
regions = load_json(region_path)
used_regions = None  # None -> all

dataset_path = "data/datasets/dataset_100k"
iteration_amount = 180_000
batch_size = 2
validation_split = 0
save_ratio = 0.05
save_dir = "save_folder"

# Optimizer config
initial_learning_rate = 1e-4
decay_steps = 1000
decay_factor = 0.95
beta_1 = 0.9
beta_2 = 0.98
weight_decay = 0.1
