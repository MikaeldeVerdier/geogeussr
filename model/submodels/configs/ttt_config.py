from data.countries import *

vocab_size = 3600 + len(COUNTRIES)  # lon/lat * 10
max_len = 3
input_shape = (max_len,)
# projection_dim = 512  # what is this?
