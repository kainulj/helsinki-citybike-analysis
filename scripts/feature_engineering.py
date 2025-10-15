"""
Feature engineering for Helsinki city bike demand modeling.

This script merges cleaned ride and weather data, aggregates hourly departures
for the top stations, and creates time-based and weather-related features.
The processed training and test datasets are saved.

Command-line arguments:
    --ride-data (str): Path to the raw bike ride CSV file.
    --weather-data (str): Path to the raw weather data CSV file.
    --output-train (str): Path to save the train feature data CSV file.
    --output-test (str): Path to save the test feature data CSV file.
    --num-station (int): Number of top stations to include based on total departures.

Example:
    python feature_engineering.py \
        --ride-data data/clean/bike_rides_cleaned.csv \
        --weather-data data/clean/weather_cleaned.csv \
        --output-train data/processed/train.csv \
        --output-test data/processed/test.csv \
        --num-stations 100
"""
import pandas as pd
import argparse
import os
from citybike.io_utils import load_csv
from citybike.features import add_features

def aggregate_hourly_departures(bike_df, num_stations):
    """
    Aggregate hourly departures for selected stations.
    Args:
        bike_df (pd.DataFrame): Dataframe containing ride data.
        num_station (int): The number of top stations included.
    Returns:
        pd.DataFrame: Hourly departures per station.
    """
    # Calculate usage per station and get top stations
    station_usage = bike_df['departure_id'].value_counts()
    top_stations = station_usage.head(num_stations).index

    # Aggregate hourly departures for top stations
    bike_df['time'] = bike_df['departure'].dt.floor('h')
    sample_bike_df = bike_df[bike_df['departure_id'].isin(top_stations)]
    bike_df_hourly = sample_bike_df.groupby(['time', 'departure_id']).size().reset_index(name='departures').rename(columns={'departure_id': 'station_id'})
    
    # Fill in missing hours with 0 rides
    season_hours = pd.date_range(start=bike_df['time'].min(), 
                        end=bike_df['time'].max(), 
                        freq='h')
    season_hours = season_hours[(season_hours.month >= 4) & (season_hours.month <= 10)]
    full_index = pd.MultiIndex.from_product([top_stations, season_hours], names=['station_id', 'time'])
    bike_df_hourly = (bike_df_hourly.set_index(['station_id', 'time'])).reindex(full_index, fill_value=0).reset_index()

    # Add station coordinates and capacities
    station_coords = bike_df[['departure_id', 'departure_lat', 'departure_lon', 'departure_capacity']].drop_duplicates()
    bike_df_hourly = bike_df_hourly.merge(station_coords, left_on='station_id', right_on='departure_id', how='left')
    bike_df_hourly.drop(columns=['departure_id'], inplace=True)

    return bike_df_hourly

def main(bike_path, weather_path, output_train, output_test, num_stations):
    """
    Main function to perform feature engineering and generate training and test datasets.
    """
    # Load cleaned data
    dtypes = {'departure_id': str, 'departure_name': str, 
            'return_id': str, 'return_name': str}
    try:
        bike_df = load_csv(bike_path, dtype=dtypes, parse_dates=['departure', 'return'])
        weather_df = load_csv(weather_path, parse_dates=['time'], index_col='time')
    except FileNotFoundError as e:
        print(e)
        return
    
    bike_df_hourly = aggregate_hourly_departures(bike_df, num_stations)

    # Merge with weather data
    bike_df_hourly = bike_df_hourly.merge(weather_df, left_on='time', right_index=True, how='left')
    bike_df_hourly.rename(
        columns={
            'departure_lat': 'lat',
            'departure_lon': 'lon',
            'departure_capacity': 'capacity'
        }, inplace=True)
    bike_df_hourly = bike_df_hourly.drop_duplicates(subset=['station_id', 'time'])
    bike_df_hourly.set_index('time', inplace=True)

    # Add lags and temporal features
    features = add_features(
        bike_df_hourly, 
        target_col='departures',
        lags=[1,2,3,6,12,18, 24, 48, 72, 168],
        same_hour_windows=[3,7],
        rolling_windows=[3, 24, 168],
        weather_cols=weather_df.columns,
        freq='h'
    )

    features.drop(columns=['time'], inplace=True)

    # Split and save
    train = features[features["year"] < 2024]
    val = features[features["year"] == 2024]
    train.to_csv(output_train, index=False)
    val.to_csv(output_test, index=False)

    print(f'Training features saved to {output_train}, testing features saved to {output_test}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ride-data', type=str, default='data/clean/bike_rides_cleaned.csv', help='Input cleaned ride data CSV file')
    parser.add_argument('--weather-data', type=str, default='data/clean/weather_cleaned.csv', help='Input cleaned weather data CSV file')
    parser.add_argument('--output-train', type=str, default='data/processed/train.csv', help='Output folder for the train feature data CSV.')
    parser.add_argument('--output-test', type=str, default='data/processed/test.csv', help='Output folder for the test feature data CSV.')
    parser.add_argument('--num-stations', type=int, default=100, help="Number of top stations to include based on total departures.")

    args = parser.parse_args()

    # Create the output folders if they don't exist
    for path in [args.output_train, args.output_test]:
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

    print(f"Using top {args.num_stations} for feature engineering.")
    main(args.ride_data, args.weather_data, args.output_train, args.output_test, args.num_stations)
