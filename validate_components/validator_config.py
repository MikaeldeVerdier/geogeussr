# Model config
image_size = (336, 336, 3)

# Testing config
regions_path = "data/geo_data.json"
used_regions = None
shapefile_path = "data/dissolved_gadm.gpkg"  # None
refinement_steps = 0

dataset_path = "data/datasets/dataset10"
shuffle = False
iteration_amount = 10

vaidation_results_path = "save_folder/validation_results.json"
inference_results_path = "save_folder/inference_results"  # None
