import random
import numpy as np

class Preprocessor:
    def __init__(self, data_augmentor=None, **kwargs):
        self.data_augmentor = data_augmentor

    def generate_description(self, location):  # could train on different prompt formats to help generalization (as data augmentation)
        city = location["coding"].get("city", None)  # really only needed for this one
        province = location["coding"].get("province", None)
        country = location["coding"].get("country", None)
        continent = location["coding"].get("continent", None)

        if city == country or city == province:
            used_city = None  # avoid "Singapore, Singapore"
        else:
            used_city = city
        if province == country:
            used_province = None
        else:
            used_province = province

        # {
        #     "continent": "Australia",
        #     "country": "Australia",
        #     "province": "New South Wales",
        #     "city": "Echuca"
        # }

        # (prompt, weight)
        available_prompts = [
            (f"A Street View photo from {country}.", 0.75),
            (f"A Street View photo from {country}, in {continent}.", 0.5),
            (f"A Street View photo from {continent}.", 0.25)
        ]
        if used_province:
            available_prompts.append((f"A Street View photo from {used_province}, {country}.", 1.25))
        if used_city:
            available_prompts.append((f"A Street View photo from {used_city}, {country}.", 1.5))
            if used_province:  # readable!
                available_prompts.append((f"A Street View photo from {used_city}, {used_province}, {country}.", 2))
        elif city is None:  # make sure there was never any city
            available_prompts.append((f"A Street View photo from rural {country}.", 1))
            if used_province:
                available_prompts.append((f"A Street View photo from rural {used_province}, {country}.", 2))

        available_prompts = np.array(available_prompts)
        used_prompt = random.choices(available_prompts[:, 0], weights=np.array(available_prompts[:, 1], dtype=np.float32))[0]

        return used_prompt

    def get_components(self, texts):  # format is so inconsistent throughout this class...
        # regions = []
        components = []
        for text in texts:
            text_comps = text.split(", latitude ")
            if "A Street View photo in " in text_comps[0]:
                region = text_comps[0].split("A Street View photo in ")[1]
                region_data = self.get_region("name", region)
                untranslated_region = region_data["code"]
            else:
                city = text_comps[0].split("A Street View photo from ")[1]
                untranslated_region = [region for region in self.regions if city in [city["name"] for city in region["cities"]]][0]["code"]

            str_latitude, str_longitude = text_comps[1].split(", longitude ")
            latitude = float(str_latitude)
            longitude = float(str_longitude)

            # regions.append(region)
            components.append([untranslated_region, latitude, longitude])

        return components

    def get_prompts(self):
        locations = [
            {
                "country": region["code"],
                "lat": region["origin"][1],
                "lng": region["origin"][0]
            }
            for region in self.regions
        ]
        prompts = []
        for location in locations:
            prompts.append(self.generate_description(location))

        return prompts

    def get_refinement_prompts_coords(self, best_prompt, refinement_amount):
        best_region_code, best_lat, best_lng = self.get_components([best_prompt])[0]
        best_region = self.get_region("code", best_region_code)
        best_region_box = best_region["bounding_box"]

        width = best_region_box[2] - best_region_box[0]
        height = best_region_box[3] - best_region_box[1]

        refinement_factor = self.refinement_base ** refinement_amount
        used_width = width * refinement_factor
        used_height = height * refinement_factor

        lowest_lat = max(best_region_box[1], best_lat - used_height / 2)  # to avoid maxing and minning could change the country if outside of box (but would then have to load gadm)
        highest_lat = min(best_region_box[3], best_lat + used_height / 2)
        lowest_lng = max(best_region_box[0], best_lng - used_width / 2)
        highest_lng = min(best_region_box[2], best_lng + used_width / 2)

        lats = np.linspace(lowest_lat, highest_lat, int(height) + 1)  # more fine guesses if the coutnry is bigger
        lngs = np.linspace(lowest_lng, highest_lng, int(width) + 1)

        locations = [
            {
                "country": best_region_code,
                "lat": lat,
                "lng": lng
            }
            for lng in lngs for lat in lats
        ]
        prompts = []
        for location in locations:
            prompts.append(self.generate_description(location))

        return prompts

    def get_refinement_prompts_city(self, best_prompt, refinement_amount):
        if refinement_amount > 0:
            return []  # can't refine more than once

        best_region_code, best_lat, best_lng = self.get_components([best_prompt])[0]
        best_region = self.get_region("code", best_region_code)
        best_region_cities = best_region["cities"]

        locations = [
            {
                "city": city["name"],
                "lat": city["origin"][1],
                "lng": city["origin"][0]
            }
            for city in best_region_cities
        ]
        prompts = []
        for location in locations:
            prompts.append(self.generate_description(location))

        return prompts
