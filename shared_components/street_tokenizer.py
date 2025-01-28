import numpy as np

class StreetTokenizer:
    def __init__(self, regions, region_translations, origins):
        self.regions = regions
        self.region_translations = region_translations
        self.origins = origins

        self.inverse_region_translations = {v: k for k, v in region_translations.items()}

    def encode_texts(self, texts):
        encoded_texts = []
        for text in texts:
            components = text.split(", ")
            country = self.region_translations[components[0]]

            encoded_texts.append(country)

        return encoded_texts

    def decode_texts(self, encoded_texts):
        texts = []
        for encoded_text in encoded_texts:
            region = self.inverse_region_translatio[encoded_text]
            region_index = self.regions.index(region)
            origin = self.origins[region_index]

            texts.append(f"{region}, latitude {origin[1]}, longitude {origin[0]}")

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
