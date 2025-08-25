"""
This script cleans the city bike data and merges it with station information.
Also cleans the weather data.
"""

import pandas as pd
import os
import argparse
from citybike.data_cleaning import merge_station_info, handle_wind_speed_gaps

def clean_ride_data(ride_data, station_data):
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

    return bike_df

def clean_weather_data(weather_data):
    # Load the weather data
    weather_df = pd.read_csv(weather_data, index_col='time')

    # Add flag column to indicate missing precipitation data and replace all missing precipitation values with -1
    weather_df['precip_missing'] = weather_df['precipitation'].isna().astype(int)
    weather_df['precipitation'] = weather_df['precipitation'].fillna(-1)

    # Forward-fill all missing temperature values
    weather_df['temperature'] = weather_df['temperature'].ffill()

    # Add flag column to indicate missing windspeed data
    weather_df['ws_missing'] = weather_df['wind_speed'].isna().astype(int)
    
    weather_df = handle_wind_speed_gaps(weather_df)

    return weather_df

def main(ride_data, station_data, weather_data, bike_output, weather_output):
    bike_df = clean_ride_data(ride_data, station_data)
    weather_df = clean_weather_data(weather_data)

    bike_df.to_csv(bike_output, index=False)
    print(f"Saved merged bike dataframe to {bike_output}")

    weather_df.to_csv(weather_output, index=True)
    print(f"Saved weather dataframe to {weather_output}")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Clean and prepare bike data")
    args.add_argument('--ride_data', type=str, default='data/raw/bike_rides.csv', help='Input ride data CSV file')
    args.add_argument('--station_data', type=str, default='data/raw/stations.csv', help='Input station data CSV file')
    args.add_argument('--weather_data', type=str, default='data/raw/weather.csv', help='Input weather data CSV file')
    args.add_argument('--bike_output', type=str, default='data/clean/bike_rides_cleaned.csv', help='Output bike ride CSV file path')
    args.add_argument('--weather_output', type=str, default='data/clean/weather_cleaned.csv', help='Output weather CSV file path')

    args = args.parse_args()

    # Create the output folders if they don't exist
    for path in [args.bike_output, args.weather_output]:
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

    main(args.ride_data, args.station_data, args.weather_data, args.bike_output, args.weather_output)