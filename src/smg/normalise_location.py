#! /usr/bin/env python3

import re
import time

from multiprocessing import Pool

import geopy
import pandas as pd


GOOGLE_API_KEY = "AIzaSyDtcYWNxzjjZRZrkPPAxEgrGD78Ey7pc50"


def preprocess_location_text(location_text):
    """
    Remove any unwanted parts from the location name. Also include bias towards
    UK locations (because that's where most of the job information is).

    :param str location_text: A string containing a location name.
    :return str: A preprocessed string suitable for geocoding.
    """
    location_text = re.sub("[^a-zA-Z\s]", " ", location_text)
    location_text = re.sub("/s+", " ", location_text)

    # when the UK is unspecified (when talking about a general region), add it
    if re.search("(north|south) (east|west)$", location_text.lower()) is not None:
        location_text += ", UK"
    elif re.search("midlands", location_text.lower()) is not None:
        location_text += ", UK"

    return location_text


def get_loc(location_text):
    """
    Performs a lookup of location_text using the Google geocoder.
    Results are then split into 3 categories - (town, region, country).

    This involves making a request for each location to the geocoder API, so
    this can take a very long time to run. There are several print statements
    that should output progress information periodically (and in case of error).

    :param str location_text: A string containing a preprocessed location name.
    :return tuple: A nested tuple containing ((town, region,country), location_text).
    """
    loc = None
    location = preprocess_location_text(location_text)

    # construct geocoder
    gc = geopy.geocoders.GoogleV3(
        api_key=GOOGLE_API_KEY,
        timeout=2
    )

    # attempt to geocode...
    try:
        loc = gc.geocode(location, region="uk")
        time.sleep(1)
    except geopy.exc.GeocoderQueryError as e:
        print("Query Error: {loc} / {e}".format(loc=location, e=e))
    except geopy.exc.GeocoderServiceError as e:
        # request failed - retry...
        try:
            loc = gc.geocode(location)
            status = "Succeeded"
            time.sleep(1)
        except:
            status = "Failed"

        message = "Service Error: {loc} / {e} / Retrying... / {status}"
        print(message.format(loc=location, e=e, status=status))

    # construct return data
    country = None
    postal_town = None
    administrative_area = None
    if loc is not None:
        for component in loc.raw["address_components"]:
            for types in component["types"]:
                if types == "country":
                    country = component["long_name"]
                if types == "administrative_area_level_2":
                    administrative_area = component["long_name"]
                if types == "postal_town":
                    postal_town = component["long_name"]

    return ((postal_town, administrative_area, country), location_text)


if __name__ == "__main__":
    df = pd.read_csv("./test.csv")

    # get unique location values from DataFrame
    raw_locations = df.groupby("LocationRaw").count().index.get_level_values(0)

    # set up batches
    list_len = len(raw_locations)
    indices = list(range(0, list_len, 1000))
    indices.append(list_len)
    with Pool(20) as p:
        locs = []
        for i in range(0, len(indices) - 1):
            location_strings = df["LocationRaw"][indices[i]:indices[i+1]]
            loc_batch = p.map(get_loc, location_strings)
            locs.extend(loc_batch)
            print("{nlocs} locations queried".format(nlocs=len(locs)))

    # add new fields to DataFrame (this is very slow)
    df["town"] = [None] * len(df["LocationRaw"])
    df["region"] = [None] * len(df["LocationRaw"])
    df["country"] = [None] * len(df["LocationRaw"])
    for loc in locs:
        town, region, country = loc[0]
        raw_location = loc[1]

        subset = df[df["LocationRaw"] == raw_location]
        subset["town"] = town
        subset["region"] = region
        subset["country"] = country
        df[df["LocationRaw"] == raw_location] = subset

    # write out new DataFrame
    df.to_csv("test_normalised_location.csv")
