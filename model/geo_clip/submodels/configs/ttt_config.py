from shared_components.files import load_json

region_path = "data/country_data.json"
regions = load_json(region_path)

vocab_size = 3601 + len(regions)  # lon/lat * 10 + 1
max_len = 3
input_shape = (max_len,)
