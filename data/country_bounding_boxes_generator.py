import json
import geopandas as gpd

from countries import COUNTRIES

shapefile_path = "data/gadm_410.gpkg"
bounding_boxes_path = "bounding_boxes.json"

if __name__ == "__main__":
    geodf = gpd.read_file(shapefile_path)
    geodf = geodf.dissolve(by="GID_0")
    # geodf = geodf.to_crs("EPSG:3857")

    bounding_boxes = []
    for country_name in COUNTRIES:
        country = geodf[geodf.index == country_name]
        minX, minY, maxX, maxY = country.geometry.total_bounds

        bounding_boxes.append([minX, minY, maxX, maxY])

    with open("bounding_boxes_path", "w") as json_file:
        json.dump(bounding_boxes, json_file)
