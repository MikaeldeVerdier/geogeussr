from shared_components.files import load_json
from model.geo_clip.submodels.configs.vit_config import input_shape
from model.geo_clip.submodels.configs.ttt_config import max_len

# Model config
image_size = input_shape
tokens_len = max_len

# Testing config
region_path = "data/country_data.json"
regions = load_json(region_path)
used_regions = None

refinement_base = 0.5
refinement_steps = 1
dataset_path = "data/datasets/dataset10"
iteration_amount = 10

test_results_path = "save_folder/test_results.json"
