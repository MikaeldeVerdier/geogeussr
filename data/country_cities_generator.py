import json
import pandas as pd

from countries import COUNTRIES, COUNTRY_TRANSLATIONS

csv_path = "data/worldcities.csv"  # this dataset isn't great, doesn't include that many cities but it's good enough, I think
cities_path = "data/country_cities.json"

top_k = 30

if __name__ == "__main__":
    df = pd.read_csv(csv_path)

    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df = df.dropna(subset=["population"])

    df_sorted = df.sort_values(by=["iso3", "population"], ascending=[True, False])
    top_cities = df_sorted.groupby("iso3").head(top_k)

    found_top_cities = []
    for country in COUNTRIES:  # these are basically iso3
        if country not in COUNTRY_TRANSLATIONS:
            print(f"Country {country} not found in the translations")
            found_top_cities.append([])
            continue

        if country not in top_cities["iso3"].values:
            print(f"Country {country} not found in the dataset")
            found_top_cities.append([])
            continue

        top_cities_country = top_cities[top_cities["iso3"] == country]
        cities = top_cities_country["city"].values.tolist()

        found_top_cities.append(cities)

    with open(cities_path, "w") as json_file:
        json.dump(found_top_cities, json_file)
