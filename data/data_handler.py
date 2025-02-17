import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import Point, MultiPoint

class DataHandler:
    def __init__(self, csv_path, gpkg_path, buffer_cities=True):
        self.mercator_crs = "EPSG:3857"
        self.geodesic_crs = "EPSG:4326"

        self.prepare_city_df(csv_path, buffer_cities)
        self.prepare_gadm_df(gpkg_path)

    def modify_city_df(self, city_df):
        city_df.loc[city_df["country"] == "Korea, South", "country"] = "South Korea"  # Name: Korea, South -> South Korea
        city_df.loc[city_df["country"] == "Korea, North", "country"] = "North Korea"  # Name: Korea, North -> North Korea
        city_df.loc[city_df["country"] == "Burma", "country"] = "Myanmar"  # Name: Burma -> Myanmar
        city_df.loc[city_df["country"] == "Islas Canarias", "country"] = "Canary Islands"  # Name: Islas Canarias -> Canary Islands
        # cities_gdf.loc[cities_gdf["country"] == "Falkland Islands (Islas Malvinas).", "country"] = "Falkland Islands"  # Name: Falkland Islands (Islas Malvinas) -> Falkland Islands
        city_df.loc[city_df["country"] == "Micronesia, Federated States of", "country"] = "Micronesia"  # Name: Micronesia, Federated States of -> Micronesia  # should it be "Federated States of Micronesia"?
        city_df.loc[city_df["country"] == "Peloponnese, Western Greece and", "country"] = "Western Greece and Peloponnese"  # Name: Peloponnese, Western Greece and -> Western Greece and Peloponnese
        city_df.loc[city_df["country"] == "Virgin Islands, British", "country"] = "British Virgin Islands"  # Name: Virgin Islands, British -> British Virgin Islands
        city_df.loc[city_df["country"] == "Bahamas, The", "country"] = "The Bahamas"  # Name: Bahamas, The -> The Bahamas
        city_df.loc[city_df["country"] == "Gambia, The", "country"] = "The Gambia"  # Name: Gambia, The -> The Gambia
        city_df.loc[city_df["country"] == "South Georgia And South Sandwich Islands", "country"] = "South Georgia and South Sandwich Islands"  # Name: South Georgia And South Sandwich Islands -> South Georgia and South Sandwich Islands  # (captilization, for some reason wrong)

        return city_df

    def estimate_radius(self, population):
        return 1000 + 3000 * np.log10(population + 1)

    def prepare_city_df(self, csv_path, buffer_cities):
        city_df = pd.read_csv(csv_path)
        city_df = gpd.GeoDataFrame(
            city_df, geometry=gpd.points_from_xy(city_df["lng"], city_df["lat"]), crs=self.geodesic_crs
        )

        # city_df = city_df.to_crs(self.mercator_crs)

        if buffer_cities:
            city_df = city_df.to_crs(self.mercator_crs)
            city_df["estimated_radius"] = city_df["population"].apply(self.estimate_radius)
            city_df["geometry"] = city_df.geometry.buffer(city_df["estimated_radius"])
            city_df = city_df.to_crs(self.geodesic_crs)

        self.city_df = self.modify_city_df(city_df)
        self.mercator_city_df = self.city_df.to_crs(self.mercator_crs)

    def modify_gadm_df(self, gadm_df):
        gadm_df.loc[gadm_df["CONTINENT"] == "Australia", "CONTINENT"] = "Oceania" # Continent: Australia -> Oceania
        # Could consider modifying countries in multiple continents ([Saint Helena, Ascension and Tristan da Cunha, Bonaire, Sint Eustatius, and Saba, Russia, Turkey, Spain, Azerbaijan, Kazakhstan])

        return gadm_df

    def prepare_gadm_df(self, gpkg_path):
        gadm_df = gpd.read_file(gpkg_path)[["GID_0", "NAME_0", "NAME_1", "CONTINENT", "geometry"]]
        if gadm_df.crs != self.geodesic_crs:
            gadm_df = gadm_df.to_crs(self.geodesic_crs)

        self.gadm_df = self.modify_gadm_df(gadm_df)
        self.mercator_gadm_df = self.gadm_df.to_crs(self.mercator_crs)

    def annotate_point(self, latitude, longitude, force_point=False):
        point = Point(longitude, latitude)
        location = self.gadm_df[self.gadm_df.contains(point)]
        if location.empty:
            if not force_point:
                return {}

            mercator_point = gpd.GeoSeries([point], crs=self.geodesic_crs).to_crs(self.mercator_crs).geometry[0]
            location = self.gadm_df.loc[self.mercator_gadm_df.distance(mercator_point).idxmin()]
        else:
            location = location.iloc[0]

        location_annotation = {
            "continent": location["CONTINENT"],
            "country": location["NAME_0"],
            "province": location["NAME_1"]  # could consider NAME_2 as well (county maybe (only available for some), would be good for us states maybe)
        }

        city_match = self.city_df[self.city_df.contains(point)]
        if not city_match.empty:
            city_name = city_match.iloc[0]["city"]
            location_annotation["city"] = city_name

        return location_annotation

    def spherically_averaged_centroid(self, geom, weights=None):
        if isinstance(geom, Point):
            return geom

        if isinstance(geom, MultiPoint):
            lng_lat = np.array([(p.x, p.y) for p in geom.geoms])
        else:
            lng_lat = geom[:, ::-1]

        lng = np.radians(lng_lat[:, 0])
        lat = np.radians(lng_lat[:, 1])

        x = np.cos(lat) * np.cos(lng)
        y = np.cos(lat) * np.sin(lng)
        z = np.sin(lat)

        if weights is None:
            weights = np.ones(len(x))

        x_mean = np.average(x, weights=weights)
        y_mean = np.average(y, weights=weights)
        z_mean = np.average(z, weights=weights)

        lng_c = np.arctan2(y_mean, x_mean)
        hyp = np.sqrt(x_mean ** 2 + y_mean ** 2)
        lat_c = np.arctan2(z_mean, hyp)

        new_point = Point(np.degrees(lng_c), np.degrees(lat_c))

        return new_point
