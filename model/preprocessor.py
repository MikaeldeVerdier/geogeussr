import numpy as np

class Preprocessor:
    def __init__(self, regions, region_origins=[], region_boxes=[], refinement_base=0.1, **kwargs):
        self.regions = regions
        self.region_origins = region_origins
        self.region_boxes = region_boxes
        self.refinement_base = refinement_base

    def generate_description(self, location):  # should be on dataset_handler but it's needed here
        return f"{location['country']}, latitude {float(location['lat']):.3f}, longitude {float(location['lng']):.3f}"

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

    def get_prompts(self):
        locations = [
            {
                "country": region,
                "lat": region_origin[1],
                "lng": region_origin[0]
            }
            for region, region_origin in zip(self.regions, self.region_origins)
        ]
        prompts = []
        for location in locations:
            prompts.append(self.generate_description(location))

        return prompts

    def get_refinement_prompts_coords(self, best_prompt, refinement_amount):
        best_region, best_lat, best_lng = self.get_components([best_prompt])[0]
        best_region_idx = self.regions.index(best_region)
        best_region_box = self.region_boxes[best_region_idx]

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
                "country": best_region,
                "lat": lat,
                "lng": lng
            }
            for lng in lngs for lat in lats
        ]
        prompts = []
        for location in locations:
            prompts.append(self.generate_description(location))

        return prompts

    """  # TODO: Implement this method
    def get_refinement_prompts_city(self, best_prompt, refinement_amount):
        best_region, best_lat, best_lng = self.get_components([best_prompt])[0]
        best_region_idx = self.regions.index(best_region)
        best_region_box = self.region_boxes[best_region_idx]

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
                "country": best_region,
                "lat": lat,
                "lng": lng
            }
            for lng in lngs for lat in lats
        ]
        prompts = []
        for location in locations:
            prompts.append(self.generate_description(location))

        return prompts
    """
