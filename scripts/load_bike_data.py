"""
This script downloads city bike ride data from the HSL API
and outputs a CSV file with the following columns:

- Departure: Departure time (datetime)
- Return: Return time (datetime)
- Departure_id: Departure station ID (string)
- Departure_name: Departure station name (string)
- Return_id: Return station ID (string)
- Return_name: Return station name (string)
- Distance: Ride distance in meters (float)
- Duration: Ride duration in seconds (float)
"""

import requests, zipfile, io
import pandas as pd
import argparse
import os

def main(start_year, end_year, output):
    dtypes = {'Departure': str, 'Return': str, 'Departure_id': str, 'Departure_name': str, 
                'Return_id': str, 'Return_name': str}
    columns = ['Departure', 'Return', 'Departure_id', 'Departure_name', 'Return_id', 'Return_name', 'Distance', 'Duration']
    dfs = []

    print(f"Downloading data from {start_year} to {end_year}")

    for year in range(start_year, end_year + 1):
        # Load the data
        r = requests.get(f'http://dev.hsl.fi/citybikes/od-trips-{year}/od-trips-{year}.zip')
        if r.status_code != 200:
            print(f"Failed to download data for year {year}: HTTP {r.status_code}")
            continue

        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            # Loop trough the files in the zip archive
            for file_name in z.namelist():
                if file_name.endswith('.csv'):
                    # Read CSV file to dataframe
                    with z.open(file_name) as f:
                        df = pd.read_csv(f,names=columns, header=0, dtype=dtypes)
                        dfs.append(df)

    # Combine the data into one DataFrame
    bike_df = pd.concat(dfs, ignore_index=True)

    # Change the departure and return columns to date time and sort the dataframe by the departure time
    bike_df['Departure'] = pd.to_datetime(bike_df['Departure'], errors='coerce')
    bike_df['Return'] = pd.to_datetime(bike_df['Return'], errors='coerce')
    bike_df.sort_values(by=['Departure'], inplace=True)

    # Remove extra whitespaces from the station names
    bike_df['Departure_name'] = bike_df['Departure_name'].str.strip()
    bike_df['Return_name'] = bike_df['Return_name'].str.strip()

    # Merge the ids for stations with identical names by assigning the most frequently used ID to each name
    # Count the occurrence of each (ID, name) pair
    station_usage = pd.concat([
        bike_df[['Departure_id', 'Departure_name']].rename(columns={'Departure_id': 'id', 'Departure_name': 'name'}),
        bike_df[['Return_id', 'Return_name']].rename(columns={'Return_id': 'id', 'Return_name': 'name'})
    ])

    usage_counts = station_usage.groupby(['id', 'name']).size().reset_index(name='count')

    # Get the most common ID for each station name
    main_ids = (
        usage_counts
        .sort_values(by=['name', 'count'], ascending=[True, False])
        .groupby(['name'])
        .first()
        .reset_index()[['name', 'id']]
        .rename(columns={'id': 'main_id'})
    )
    # Merge the main IDs back to the usage counts
    id_map = usage_counts.merge(main_ids, on='name')

    # Create a dictionary to map old IDs to main IDs
    id_merge_dict = id_map.set_index('id')['main_id'].to_dict()

    # Map the old IDs to main IDs in the bike_df
    bike_df['Departure_id'] = bike_df['Departure_id'].map(id_merge_dict).fillna(bike_df['Departure_id'])
    bike_df['Return_id'] = bike_df['Return_id'].map(id_merge_dict).fillna(bike_df['Return_id'] )

    # Some station names might have changed over time, so we update the names to the latest known names
    # Get the latest name per Departure_id
    dep_names = (
        bike_df.dropna(subset=['Departure_name'])
        .drop_duplicates('Departure_id', keep='last')
        .set_index('Departure_id')['Departure_name']
    )
    # Get the latest name per Return_id
    ret_names = (
        bike_df.dropna(subset=['Return_name'])
        .drop_duplicates('Return_id', keep='last')
        .set_index('Return_id')['Return_name']
    )

    # Map the latest names to match the station ids
    station_dict = {**ret_names.to_dict(), **dep_names.to_dict()}
    bike_df['Departure_name'] = bike_df['Departure_id'].map(station_dict)
    bike_df['Return_name'] = bike_df['Return_id'].map(station_dict)

    # Change the column names to lower case
    bike_df.columns = bike_df.columns.str.lower()

    bike_df.to_csv(output, index=False)
    print(f"Saved bike ride data to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download data script")

    parser.add_argument('--start-year', type=int, default=2020, help='Start year of data range')
    parser.add_argument('--end-year', type=int, default=2024, help='End year of data range')
    parser.add_argument('--output', type=str, default='data/raw/bike_rides.csv', help='Output CSV file path')

    args = parser.parse_args()

    # Create the output folder if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    main(args.start_year, args.end_year, args.output)
