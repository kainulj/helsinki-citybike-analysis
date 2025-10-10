"""
Fetch Helsinki city bike ride data from the HSL API and save it as a CSV file.

This script retrieves bike ride data for the specified year range from the HSL open data API,
combines all records into a single dataset, and outputs a CSV file with the following columns:

Output columns:
- departure: Departure time (datetime)
- return: Return time (datetime)
- departure_id: Departure station ID (string)
- departure_name: Departure station name (string)
- return_id: Return station ID (string)
- return_name: Return station name (string)
- distance: Ride distance in meters (float)
- duration: Ride duration in seconds (float)

Command-line arguments:
    --start-year (int): The first year of the timeframe (inclusive).
    --end-year (int): The last year of the timeframe (inclusive).
    --output (str): Path to save the processed CSV file.

Example:
    python fetch_bike_data.py \
        --start-year 2020 \
        --end-year 2024 \
        --output data/raw/bike_rides.csv
"""

import requests, zipfile, io
import pandas as pd
import argparse
import os

def fetch_data(start_year, end_year):
    """
    Fetch city bike data from HSL API.
    Args:
        start_year (int): The first year of the time range.
        end_year (int): The last year of the time range.
    Returns:
        pd.DataFrame: DataFrame with the fetched data.
    """
    dtypes = {'Departure': str, 'Return': str, 'Departure_id': str, 'Departure_name': str, 
                'Return_id': str, 'Return_name': str}
    columns = ['Departure', 'Return', 'Departure_id', 'Departure_name', 'Return_id', 'Return_name', 'Distance', 'Duration']
    dfs = []

    print(f"Fetching data from {start_year} to {end_year}")

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
    return bike_df

def merge_station_ids(df):
    """
    Merge IDs for stations with identical names by assigning the most frequently used ID to each name.
    Args:
        df (pd.DataFrame): DataFrame with ride data.
    Returns:
        pd.DataFrame: DataFrame with merged station IDs.
    """

    # Count the occurrence of each (ID, name) pair
    station_usage = pd.concat([
        df[['departure_id', 'departure_name']].rename(columns={'departure_id': 'id', 'departure_name': 'name'}),
        df[['return_id', 'return_name']].rename(columns={'return_id': 'id', 'return_name': 'name'})
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

    id_map = usage_counts.merge(main_ids, on='name')

    # Create a dictionary to map old IDs to main IDs
    id_merge_dict = id_map.set_index('id')['main_id'].to_dict()

    # Map the old IDs to main IDs in the bike_df
    df['departure_id'] = df['departure_id'].map(id_merge_dict).fillna(df['departure_id'])
    df['return_id'] = df['return_id'].map(id_merge_dict).fillna(df['return_id'])
    return df

def update_station_names(df):
    """
    Update station names to the latest known names for each station ID.
    Args:
        df (pd.DataFrame): DataFrame with ride data.
    Returns:
        pd.DataFrame: DataFrame with updated station names.
    """
    
    # Get the latest known departure names per station ID
    dep_names = (
        df.dropna(subset=['departure_name'])
        .drop_duplicates('departure_id', keep='last')
        .set_index('departure_id')['departure_name']
    )

    # Get the latest known return names per station ID
    ret_names = (
        df.dropna(subset=['return_name'])
        .drop_duplicates('return_id', keep='last')
        .set_index('return_id')['return_name']
    )

    # Map the latest names to match the station ids
    station_dict = {**ret_names.to_dict(), **dep_names.to_dict()}
    df['departure_name'] = df['departure_id'].map(station_dict)
    df['return_name'] = df['return_id'].map(station_dict)
    return df

def main(start_year, end_year, output):
    """
    Main pipeline for processing raw bike trip data.
    Fetches, cleans, merges station info, and saves the result to CSV.
    Args:
        start_year (int): The start year of the time range.
        end_year (int): The end year of the time range.
        output (str): Path to output CSV file.
    """
    bike_df = fetch_data(start_year, end_year)

    # Change the column names to lower case
    bike_df.columns = bike_df.columns.str.lower()

    # Convert departure and return times to datetime, then sort by departure time
    bike_df['departure'] = pd.to_datetime(bike_df['departure'], errors='coerce')
    bike_df['return'] = pd.to_datetime(bike_df['return'], errors='coerce')
    bike_df.sort_values(by=['departure'], inplace=True)
    
    # Clean the station names
    bike_df['departure_name'] = bike_df['departure_name'].str.strip()
    bike_df['return_name'] = bike_df['return_name'].str.strip()

    bike_df = merge_station_ids(bike_df)
    bike_df = update_station_names(bike_df)

    bike_df.to_csv(output, index=False)
    print(f"Saved bike ride data to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download data script")

    parser.add_argument('--start-year', type=int, default=2020, help='Start year of data range')
    parser.add_argument('--end-year', type=int, default=2024, help='End year of data range')
    parser.add_argument('--output', type=str, default='data/raw/bike_rides.csv', help='Output CSV file path')

    args = parser.parse_args()

    # Create the output folder if it doesn't exist
    dir_name = os.path.dirname(args.output)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    main(args.start_year, args.end_year, args.output)