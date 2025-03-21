# Model config
image_size = (224, 224, 3)

# Testing config
regions_path = "data/geo_data.json"
used_regions = None
gadm_path = "data/gadm_410.gpkg"  # None
city_path = "data/worldcities.csv"  # None
refinement_steps = 1

dataset_path = "data/datasets/dataset10"
shuffle = False
iteration_amount = 10

vaidation_results_path = "save_folder/validation_results.json"
inference_results_path = "save_folder/inference_results"  # None
