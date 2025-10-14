"""
Generate all exploratory data analysis (EDA) visualizations for Helsinki city bike data.

This script runs all individual plotting modules to produce visualizations of
temporal trends, spatial patterns, trip distributions, and weather relationships.
Figures are saved to their respective subdirectories under 'figures/'.
"""
from plot_distributions import create_distribution_plots
from plot_spatial_patterns import create_spatial_plots
from plot_weather_relations import create_weather_plots
from plot_temporal_relations import create_temporal_plots
from citybike.io_utils import load_csv
from pathlib import Path

BIKE_PATH = Path('data/clean/bike_rides_cleaned.csv')
WEATHER_PATH = Path('data/clean/weather_cleaned.csv')

def main():
    """
    Main function to generate all EDA visualizations for city bike data analysis.
    """

    # Load the data
    dtypes = {'departure_id': str, 'departure_name': str, 
            'return_id': str, 'return_name': str}
    try:
        bike_df = load_csv(BIKE_PATH, dtype=dtypes, parse_dates=['departure', 'return'])
        weather_df = load_csv(WEATHER_PATH, parse_dates=['time'], index_col='time')
    except FileNotFoundError as e:
        print(e)
        return
    
    create_distribution_plots(bike_df)
    create_spatial_plots(bike_df)
    create_weather_plots(bike_df, weather_df)
    create_temporal_plots(bike_df)

if __name__ == "__main__":
    main()
