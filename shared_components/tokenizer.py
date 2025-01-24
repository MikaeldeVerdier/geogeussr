import numpy as np

class Tokenizer:
    def __init__(self, regions):
        self.regions = regions

    def encode_texts(self, texts):
        encoded_texts = []
        for text in texts:
            components = text.split(", ")
            country = components[0]
            latitude = float(components[1].split(" ")[1])
            longitude = float(components[2].split(" ")[1])

            country_idx = self.regions.index(country)
            lat_norm = (latitude + 90) / 180  # Normalize to [0, 1]
            lng_norm = (longitude + 180) / 360  # Normalize to [0, 1]

            encoded_texts.append([country_idx, lat_norm, lng_norm])

        return np.array(encoded_texts)

    def decode_texts(self, encoded_texts):
        texts = []
        for encoded_text in encoded_texts:
            country = self.regions[int(encoded_text[0])] if encoded_text[0] != -1 else "Unknown"
            latitude = encoded_text[1] * 180 - 90
            longitude = encoded_text[1] * 360 - 180

            texts.append(f"{country}, latitude {latitude}, longitude {longitude}")

        return np.array(texts)
