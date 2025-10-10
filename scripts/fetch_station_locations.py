"""
Fetch city bike station locations from Digitransit and OpenStreetMap (OSM) and save them as a CSV file.

This script retrieves official HSL city bike stations from the Digitransit API and
supplements missing or unmatched entries with data from OpenStreetMap.
An API key is required to access Digitransit data. Only OSM stations will be fetched if API is not provided.

Output columns:
- ID: Station identifier (string; may be missing for OSM stations)
- name: Station name (string)
- lat: Latitude (float)
- lon: Longitude (float)
- capacity: Number of bikes the station can hold (integer)
- source: Data source ('HSL' or 'OSM')

The script uses the Digitransit API to fetch official HSL bike stations
and supplements missing entries using OpenStreetMap.
An API key is required to access data from Digitransit without it the script only fetches OSM data.

Command-line arguments:
    --output (str): Path to save the processed CSV file.
    --api-key (str): Digitransit API key. If omitted, only OSM data will be downloaded.

Example:
    python fetch_station_data.py \
        --output data/raw/stations.csv
        --api-key <YOUR_API_KEY>
"""

import overpy
import pandas as pd
import argparse
import requests
import os

def fetch_digitransit_stations(api_key):
    """
    Fetch city bike station data from Digitransit API using the provided API key.
    Args:
        api_key (str): Digitransit API key.
    Returns:
        pd.DataFrame: DataFrame containing station ID, name, lat, lon, capacity, and source.
    """

    url = "https://api.digitransit.fi/routing/v2/hsl/gtfs/v1"

    query = """
    {
    vehicleRentalStations {
        stationId
        name
        lat
        lon
        capacity
    }
    }
    """

    headers = {
        "Content-Type": "application/json",
        "digitransit-subscription-key": api_key
    }

    response = requests.post(url, json={"query": query}, headers=headers)

    # Extract data from the result
    if response.status_code == 200:
        data = response.json()
        stations = data["data"]["vehicleRentalStations"]
        hsl_df = pd.DataFrame(stations)
        hsl_df.rename(columns={"stationId": "ID"}, inplace=True)

    # Keep stations in Espoo and Helsinki
    ID_PREFIX = 'smoove:'

    # Filter by known station ID prefix and remove it from the IDs
    hsl_df = hsl_df[hsl_df['ID'].str.startswith(ID_PREFIX)]
    hsl_df['ID'] = hsl_df['ID'].str.replace(ID_PREFIX, '', regex=False)

    hsl_df['source'] = 'HSL'

    return hsl_df

def fetch_OSM_stations():
    """
    Fetch city bike station data from OpenStreetMap.
    Returns:
        pd.DataFrame: DataFrame containing station name, lat, lon, capacity, and source.
    """

    api = overpy.Overpass()
    
    query = """
    area["name"="Helsinki"]->.helsinki;
    area["name"="Espoo"]->.espoo;

    (
    node["amenity"="bicycle_rental"](area.helsinki);
    node["amenity"="bicycle_rental"](area.espoo);
    );
    out body;
    """

    result = api.query(query)

    # Extract data from the result
    data = []
    for node in result.nodes:
        data.append({
            "name": node.tags.get("name", "Unknown"),
            "lat": float(node.lat),
            "lon": float(node.lon),
            "capacity": int(node.tags.get("capacity", 0))
        })


    osm_df = pd.DataFrame(data)
    osm_df['source'] = 'OSM'

    # Remove rows with 'Unknown' name
    osm_df = osm_df[osm_df['name'] != 'Unknown']
    return osm_df

def main(output, api_key):
    """
    Main function to fetch station data from Digitransit and OSM, merge, and save to CSV.
    Args:
        output (str): Path to output CSV file.
        api_key (str): Digitransit API key.
    """
    if api_key:
        hsl_df = fetch_digitransit_stations(api_key)
        osm_df = fetch_OSM_stations()
    
        # Use HSL data as the base and add missing stations from OSM
        matched_names = set(hsl_df['name'])
        osm_missing = osm_df[~osm_df['name'].isin(matched_names)].copy()
        osm_missing['ID'] = ''

        df = pd.concat([hsl_df, osm_missing], ignore_index=True)
    else:
        print('API key not provided, using OSM data')
        df = fetch_OSM_stations()
        df['ID'] = ''

    # Change the column names to lower case
    df.columns = df.columns.str.lower()

    df.to_csv(args.output, index=False)
    print(f"Saved station locations to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch station locations script")

    parser.add_argument('--output', type=str, default='data/raw/stations.csv', help='Output CSV file path')
    parser.add_argument('--api-key', type=str, default='', help='API key for Digitransit')

    args = parser.parse_args()

    # Create the output folder if it doesn't exist
    dir_name = os.path.dirname(args.output)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    main(args.output, args.api_key)