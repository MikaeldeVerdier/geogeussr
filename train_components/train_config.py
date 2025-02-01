from data.countries import *
from model.geo_clip.submodels.configs.vit_config import input_shape
from model.geo_clip.submodels.configs.ttt_config import max_len

# Model config
image_size = input_shape
tokens_len = max_len

# Training config
regions = COUNTRIES
used_regions = None
region_translations = COUNTRY_TRANSLATIONS
region_origins = COUNTRY_ORIGINGS
dataset_path = "data/datasets/dataset10"
iteration_amount = 10
batch_size = 8
validation_split = 0.2
save_ratio = 0.5
save_dir = "save_folder"

# Optimizer config
initial_learning_rate = 1e-6
decay_steps = 1000
decay_factor = 0.95
beta_1 = 0.9
beta_2 = 0.98
weight_decay = 0.1
