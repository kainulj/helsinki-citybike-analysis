"""
This script downloads city bike station locations from Digitransit and OpenStreetMap,
and outputs a CSV file with the following columns:

- ID: Station identifier (string; may be missing for OSM stations)
- name: Station name (string)
- lat: Latitude (float)
- lon: Longitude (float)
- capacity: Number of bikes the station can hold (integer)
- source: Data source ('HSL' or 'OSM')

The script uses the Digitransit API to fetch official HSL bike stations
and supplements missing entries using OpenStreetMap.
An API key is required to access data from Digitransit.
"""

import overpy
import pandas as pd
import argparse
import requests
from pathlib import Path

DATA_DIR = Path('data/raw/')

parser = argparse.ArgumentParser(description="Fetch station locations script")

parser.add_argument('--output', type=str, default='station_locations.csv', help='Output file name')
parser.add_argument('--api-key', type=str, required=True, help='API key for Digitransit')

args = parser.parse_args()

api = overpy.Overpass()

print("Fetching station locations from Digitransit...")
# Fetch from Digitransit API
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
    "digitransit-subscription-key": args.api_key
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

print("Fetching station locations from OpenStreetMap...")
# Fetch from OpenStreetMap
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

# Use HSL data as the base and add missing stations from OSM
matched_names = set(hsl_df['name'])
osm_missing = osm_df[~osm_df['name'].isin(matched_names)].copy()
osm_missing['ID'] = ''

df = pd.concat([hsl_df, osm_missing], ignore_index=True)

# Change the column names to lower case
df.columns = df.columns.str.lower()

# Create the data folder if it doesn't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"Saving to {DATA_DIR / args.output}")
df.to_csv(DATA_DIR / args.output, index=False)