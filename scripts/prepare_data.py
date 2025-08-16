"""
This script cleans the city bike data and merges it with station information.
"""

import pandas as pd
import os
import argparse


def merge_station_info(bike_df, station_df, station_type):
    """ Merges station information into the bike ride DataFrame. """

    hsl_stations = station_df[station_df['source'] == 'HSL']
    osm_stations = station_df[station_df['source'] == 'OSM']

    # Merge HSL station info
    bike_df = bike_df.merge(
        hsl_stations[['lat', 'lon', 'capacity']].add_prefix(f'{station_type}_'),
        left_on=f'{station_type}_id',
        right_index=True,
        how='left'
    )
    # Merge OSM station info as separate columns
    bike_df = bike_df.merge(
        osm_stations[['lat', 'lon', 'capacity', 'name']].add_prefix(f'OSM_{station_type}_'),
        left_on=f'{station_type}_name',
        right_on=f'OSM_{station_type}_name',
        how='left'
    )

    # Fill missing values from OSM
    for col in ['lat', 'lon', 'capacity']:
        bike_df[f'{station_type}_{col}'] = bike_df[f'{station_type}_{col}'].fillna(bike_df[f'OSM_{station_type}_{col}'])

    # Drop the OSM columns
    bike_df = bike_df.drop(columns=[
        f'OSM_{station_type}_name',
        f'OSM_{station_type}_lat',
        f'OSM_{station_type}_lon',
        f'OSM_{station_type}_capacity'
    ])

    return bike_df

def main(ride_data, station_data, output):
    # Load the bike ride data
    dtypes = {'departure_id': str, 'departure_name': str, 
            'return_id': str, 'return_name': str}
    bike_df = pd.read_csv(ride_data, dtype=dtypes, parse_dates=['departure', 'return'])

    # Load the station data
    station_df = pd.read_csv(station_data)
    station_df = station_df.set_index('id')
    # Add leading zeros to IDs
    station_df.index = station_df.index.fillna(-1).astype(int).astype(str).str.zfill(3)

    # Remove rides from March to keep the dataset consistent
    bike_df = bike_df[bike_df['departure'].dt.month != 3]

    bike_df['duration_calc'] = (
    pd.to_datetime(bike_df['return']) - pd.to_datetime(bike_df['departure'])
        ).dt.total_seconds()

    bike_df['duration_diff'] = abs(bike_df['duration_calc'] - bike_df['duration'])

    # Fill the missing duration values with calculated durations
    bike_df.loc[bike_df['duration'].isna(), 'duration'] = bike_df.loc[bike_df['duration'].isna(), 'duration_calc']

    # Drop rows with missing values and temporary duration columns
    bike_df = bike_df.drop(columns=['duration_calc', 'duration_diff']).dropna()

    # Convert seconds to minutes
    bike_df['duration'] = bike_df['duration'] / 60

    # Remove outliers based on duration and distance
    bike_df = bike_df[(bike_df['duration'] <= 5 * 60) & (bike_df['duration'] > 1) & (bike_df['distance'] > 50) & (bike_df['distance'] < 15000)]

    # Merge with station information
    bike_df = merge_station_info(bike_df, station_df, station_type='departure')
    bike_df = merge_station_info(bike_df, station_df, station_type='return')

    bike_df.to_csv(output, index=False)
    print(f"Saved merged dataframe to {output}")

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Clean and prepare bike data")
    args.add_argument('--ride_data', type=str, default='data/raw/bike_rides.csv', help='Input ride data CSV file')
    args.add_argument('--station_data', type=str, default='data/raw/stations.csv', help='Input station data CSV file')
    args.add_argument('--output', type=str, default='data/clean/cleaned_bike_data.csv', help='Output CSV file path')

    args = args.parse_args()

    # Create the output folder if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    main(args.ride_data, args.station_data, args.output)