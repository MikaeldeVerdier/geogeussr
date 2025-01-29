from data.countries import *
from model.geo_clip.submodels.configs.vit_config import input_shape
from model.geo_clip.submodels.configs.ttt_config import max_len

# Model config
image_size = input_shape
tokens_len = max_len

# Testing config
regions = COUNTRIES
used_regions = None
region_translations = COUNTRY_TRANSLATIONS
region_origins = COUNTRY_ORIGINGS
dataset_path = "data/datasets/dataset10"
iteration_amount = 10
