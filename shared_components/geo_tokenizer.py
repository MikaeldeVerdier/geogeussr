import numpy as np

class GeoTokenizer:
    def __init__(self, regions, *args):
        self.regions = regions

    def encode_texts(self, texts):
        encoded_texts = []
        for text in texts:
            components = text.split(", ")
            country = components[0]
            latitude = float(components[1].split(" ")[1])
            longitude = float(components[2].split(" ")[1])

            country_idx = self.regions.index(country) + 3600
            encoded_lat = np.round(latitude + 90, 1) * 10  # in the range [0, 1800]
            encoded_lng = np.round(longitude + 180, 1) * 10  # in the range [0, 3600]

            encoded_texts.append([country_idx, encoded_lat, encoded_lng])

        return np.array(encoded_texts)

    def decode_texts(self, encoded_texts):
        texts = []
        for encoded_text in encoded_texts:
            country = self.regions[int(encoded_text[0] - 3600)] if encoded_text[0] != -1 else "Unknown"
            latitude = encoded_text[1] / 10
            longitude = encoded_text[1] / 10

            texts.append(f"{country}, latitude {latitude}, longitude {longitude}")

        return np.array(texts)

    def get_components(self, texts):  # format is so inconsistent throughout this class...
        # regions = []
        components = []
        for text in texts:
            text_comps = text.split(", ")
            region = text_comps[0]
            latitude = float(text_comps[1].split(" ")[1])
            longitude = float(text_comps[2].split(" ")[1])

            # regions.append(region)
            components.append([region, latitude, longitude])

        return components
