import overpy
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Fetch station locations script")

parser.add_argument('--output', type=str, default='station_locations.csv', help='Output file name')

args = parser.parse_args()

api = overpy.Overpass()

print("Fetching station locations...")

# Query to find bicycle station in Helsinki and Espoo
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

# Extracting data from the result
data = []
for node in result.nodes:
    data.append({
        "name": node.tags.get("name", "Unknown"),
        "lat": float(node.lat),
        "lon": float(node.lon)
    })

# Convert to DataFrame
df = pd.DataFrame(data)

print(f"Saving to {args.output}")
df.to_csv(args.output, index=False)

