class Preprocessor:
    def __init__(self, data_augmentor=None, **kwargs):
        self.data_augmentor = data_augmentor

    def generate_description(self, location):  # could train on different prompt formats to help generalization (as data augmentation)        
        city = location["coding"].get("city", None) or None  # really only needed for this one
        province = location["coding"].get("province", None) or None
        country = location["coding"].get("country", None) or None
        continent = location["coding"].get("continent", None) or None  # "or None" to handle empty strings

        if city is not None:  # gotta handle other cases somehow
            return f"A Street View photo from {city}, {province}, {country}, in {continent}."
        else:
            return f"A Street View photo from rural {province}, {country}, in {continent}."

        """  # should this be used?
        if city == country or city == province:
            used_city = None  # avoid "Singapore, Singapore"
        else:
            used_city = city
        if province == country:
            used_province = None
        else:
            used_province = province

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
        """

    def get_basic_descriptions(self, continent=None, country=None, province=None, city=None, use_all=True):
        available_prompts = [
            ("A Street View photo from {city}, {province}, {country}.", ["city", "province", "country"]),
            ("A Street View photo from {city}, {country}.", ["city", "country"]),
            ("A Street View photo from {city}, {province}.", ["city", "province"]),
            ("A Street View photo from {city}.", ["city"]),  # for Singapore and such
            ("A Street View photo from {province}, {country}.", ["province", "country"]),
            ("A Street View photo from {country}, in {continent}.", ["country", "continent"]),
            ("A Street View photo from {country}.", ["country"]),
            ("A Street View photo from {continent}.", ["continent"]),
            # ("A Street View photo from {city}, {province}, {country}, in {continent}.", ["city", "province", "country", "continent"]),
        ]

        format_dict = {}
        if city:
            format_dict["city"] = city
        if province and province not in format_dict.values():
            format_dict["province"] = province
        if country and country not in format_dict.values():
            format_dict["country"] = country
        if continent and continent not in format_dict.values():
            format_dict["continent"] = continent

        filtered_available_prompts = []
        for prompt, required_keys in available_prompts:
            format_has_all = all([required_key in format_dict.keys() for required_key in required_keys])
            has_all_format = all([key in required_keys for key in format_dict.keys()])
            if format_has_all and (has_all_format or not use_all):
                filtered_available_prompts.append(prompt.format(**format_dict))

        return filtered_available_prompts

    def get_rural_descriptions(self, continent=None, country=None, province=None, use_all=True):
        available_prompts = [
            ("A Street View photo from rural {country}.", ["country"]),
            ("A Street View photo from rural {province}, {country}.", ["province", "country"]),
            ("A Street View photo from rural {continent}.", ["continent"])
        ]

        format_dict = {}
        if province and province not in format_dict.values():
            format_dict["province"] = province
        if country and country not in format_dict.values():
            format_dict["country"] = country
        if continent and continent not in format_dict.values():
            format_dict["continent"] = continent

        filtered_available_prompts = []
        for prompt, required_keys in available_prompts:
            format_has_all = all([required_key in format_dict.keys() for required_key in required_keys])
            has_all_format = all([key in required_keys for key in format_dict.keys()])
            if format_has_all and (has_all_format or not use_all):
                filtered_available_prompts.append(prompt.format(**format_dict))

        return filtered_available_prompts

    """
    # implement for other refinement levels
    def get_components(self, texts, regions):
        components = []
        for text in texts:
            text_comps = text.split("A Street View photo from ")[1].split(", ")
            country = text_comps[-1].split(".")[0]

            for region in regions.keys():
                if country in regions[region]["countries"]:
                    continent = region
                    origin = regions[region]["countries"][country]["origin"][::-1]  # to (lat, lng)

                    break

            # regions.append(region)
            components.append([[continent, country], origin])

        return components
    """

    def get_prompts(self, regions, refinement_args=[], refinement_amount=1):
        refinement_args += [None] * (refinement_amount - len(refinement_args) + 1)

        prompts = []
        prompt_components = []
        for continent, countries in regions.items():  # could do recursively instead
            if refinement_args[0] is not None and refinement_args[0] != continent:
                continue

            continent_origin = countries["origin"][::-1]  # to (lat, lng)

            if refinement_amount == 0:
                prompt = [p for p in self.get_basic_descriptions(continent=continent) if p not in prompts]  # continent only used here (should it be used later too?)
                prompts += prompt
                prompt_components += [[[continent], continent_origin]] * len(prompt)
                continue

            for country, provinces in countries["countries"].items():
                if refinement_args[1] is not None and refinement_args[1] != country:
                    continue

                country_origin = provinces["origin"][::-1]

                if refinement_amount == 1:
                    prompt = [p for p in self.get_basic_descriptions(country=country) if p not in prompts]
                    prompts += prompt
                    prompt_components += [[[continent, country], country_origin]] * len(prompt)
                    continue

                for province, cities in provinces["provinces"].items():
                    if refinement_args[2] is not None and refinement_args[2] != province:
                        continue

                    province_origin = cities["origin"][::-1]

                    if refinement_amount == 2:
                        prompt = [p for p in self.get_basic_descriptions(country=country, province=province) if p not in prompts]
                        prompts += prompt
                        prompt_components += [[[continent, country, province], province_origin]] * len(prompt)
                        continue

                    prompt = [p for p in self.get_rural_descriptions(country=country, province=province) if p not in prompts]
                    prompts += prompt
                    prompt_components += [[[continent, country, province], province_origin]] * len(prompt)
                    for city in cities["cities"]:
                        city_name = city["name"]

                        if refinement_args[3] is not None and refinement_args[3] != city_name:
                            continue

                        city_origin = city["origin"][::-1]

                        prompt = [p for p in self.get_basic_descriptions(country=country, province=province, city=city_name) if p not in prompts]
                        prompts += prompt
                        prompt_components += [[[continent, country, province, city_name], city_origin]] * len(prompt)

        return prompts, prompt_components
