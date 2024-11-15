import json
import geopandas as gpd

from countries import *

SHAPEFILE_PATH = "dataset_generator/gadm_410.gpkg"
ORIGINS_PATH = "country_origins.json"

if __name__ == "__main__":
    geodf = gpd.read_file(SHAPEFILE_PATH)
    geodf = geodf.dissolve(by="GID_0")
    geodf = geodf.to_crs("EPSG:3857")

    origins = []
    for country_name in COUNTRIES:
        country = geodf[geodf.index == country_name]
        origin = country.geometry.centroid.to_crs("EPSG:4326")._values[0]  # would maybe be faster to save geometry and then call to_crs after on a geoDF of all of them

        origins.append([origin.x, origin.y])

    with open(ORIGINS_PATH, "w") as json_file:
        json.dump(origins, json_file)
