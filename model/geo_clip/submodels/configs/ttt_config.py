from data.countries import *

vocab_size = 3601 + len(COUNTRIES)  # lon/lat * 10 + 1
max_len = 3
input_shape = (max_len,)
