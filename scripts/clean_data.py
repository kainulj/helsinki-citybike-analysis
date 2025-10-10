"""
Clean and preprocess Helsinki city bike ride and weather data for analysis.

This script performs data cleaning and preprocessing for the Helsinki city bike dataset.
It prepares cleaned versions of both ride and weather data for downstream analysis and modeling.

Features:
- Cleans raw bike ride data, removes outliers, fills missing values, and merges with station metadata.
- Cleans weather data, handles missing values and gaps, and adds flags for missing measurements.
- Outputs cleaned bike ride and weather datasets as CSV files for downstream analysis.

Inputs:
- Raw bike ride CSV file (with ride details and station IDs)
- Station metadata CSV file (with station names, locations, and IDs)
- Raw weather CSV file (with temperature, wind speed, precipitation)

Outputs:
- Cleaned bike ride CSV file
- Cleaned weather CSV file

Command-line arguments:
    --ride-data (str): Path to the raw bike ride CSV file.
    --station-data (str): Path to the station metadata CSV file.
    --weather-data (str): Path to the raw weather data CSV file.
    --bike-output (str): Path to save the cleaned bike ride data CSV file.
    --weather-output (str): Path to save the cleaned weather data CSV file.

Example:
    python clean_data.py \
        --ride-data data/raw/bike_rides.csv \
        --station-data data/raw/stations.csv \
        --weather-data data/raw/weather.csv \
        --bike-output data/clean/bike_rides_cleaned.csv \
        --weather-output data/clean/weather_cleaned.csv
"""

import pandas as pd
import os
import argparse
from citybike.data_cleaning import merge_station_info, handle_wind_speed_gaps

def clean_ride_data(ride_data, station_data):
    """
    Clean bike ride data and merge with station information.
    Args:
        ride_data (str): Path to raw bike ride CSV file.
        station_data (str): Path to station info CSV file.
    Returns:
        pd.DataFrame: Cleaned and merged bike ride dataframe.
    """
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
    """
    Clean weather data by handling missing values and gaps.
    Args:
        weather_data (str): Path to raw weather CSV file.
    Returns:
        pd.DataFrame: Cleaned weather dataframe.
    """
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
    """
    Main function to clean bike and weather data, then save to output files.
    Args:
        ride_data (str): Path to raw bike ride CSV file.
        station_data (str): Path to station info CSV file.
        weather_data (str): Path to raw weather CSV file.
        bike_output (str): Output path for cleaned bike data.
        weather_output (str): Output path for cleaned weather data.
    """
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