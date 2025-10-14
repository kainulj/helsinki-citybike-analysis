"""
Weather analysis and visualization for Helsinki city bike usage.

This script examines how weather conditions affect bike ride departures.
It generates a correlation matrix and scatter plots illustrating relationships
between temperature, wind speed, precipitation, and departure counts.

All figures are saved to the 'figures/weather' directory.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from citybike.io_utils import load_csv

# Style settings
sns.set_theme(context='paper',style='white')
palette = sns.color_palette("Dark2")
sns.set_palette(palette)
plt.rcParams["figure.figsize"] = (8, 5)

# Paths
BIKE_PATH = Path("data/clean/bike_rides_cleaned.csv")
WEATHER_PATH = Path("data/clean/weather_cleaned.csv")
OUTPUT_DIR = Path("figures/weather")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_weather_corr_matrix(df):
    """
    Plot and save a correlation matrix of weather features and departures.
    Args:
        df (pd.DataFrame): Dataframe containing weather and ride data.
    """
    # Omit rows with missing wind speed or precipitation
    df[['wind_speed', 'precipitation']] = df[['wind_speed', 'precipitation']].replace(-1, np.nan)

    # Create the correlation matrix
    corr = df[['departures', 'temperature', 'wind_speed', 'precipitation']].corr()
    corr.columns = [col.title() for col in corr.columns]
    plt.figure()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation of Weather Features with Departure Count')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'weather_corr_matrix.png')
    plt.close()

def plot_weather_scatter(df):
    """
    Plot and save scatter plots of temperature, wind speed, and precipitation vs departure count
    Args:
        df (pd.DataFrame): Dataframe containing weather and ride data.
        path (str): Directory to save plot.
    """
    # Temperature vs departure count
    plt.figure()
    sns.scatterplot(x='temperature', y='departures', data=df)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Number of Departures')
    plt.title('Temperature’s Effect on Bike Usage')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'scatter_temperature.png')
    plt.close()

    # Wind speed vs departure count
    plt.figure()
    sns.scatterplot(x='wind_speed', y='departures', data=df[df['ws_missing'] == 0])
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Number of Departures')
    plt.title('Wind Speed’s Effect on Bike Usage')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'scatter_windspeed.png')
    plt.close()

    # Precipitation vs departure count
    plt.figure()
    sns.scatterplot(x='precipitation', y='departures', data=df[df['precip_missing'] == 0])
    plt.xlabel('Precipitation (mm)')
    plt.ylabel('Number of Departures')
    plt.title('Precipitation’s Effect on Bike Usage')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'scatter_precipitation.png')
    plt.close()

def create_weather_plots(bike_df, weather_df):
    """
    Creates weather-related plots.
    Args:
        bike_df (pd.DataFrame): Dataframe containing ride data.
        weather_df (pd.Dataframe): Dataframe containing weather data.
    """

    # Calculate hourly departures
    bike_df['time'] = bike_df['departure'].dt.floor('h')
    bike_usage = bike_df.groupby(['time']).size().reset_index(name='departures')
    
    season_hours = pd.date_range(start=bike_df['time'].min(), 
                        end=bike_df['time'].max(), 
                        freq='h')
    season_hours = season_hours[(season_hours.month >= 4) & (season_hours.month <= 10)]
    bike_usage = (bike_usage.set_index(['time'])).reindex(season_hours, fill_value=0)

    # Merge the dataframes
    df = pd.merge_asof(bike_usage, weather_df, left_index=True, right_index=True, tolerance=pd.Timedelta(1,'m'))

    # Create the plots
    plot_weather_corr_matrix(df)
    plot_weather_scatter(df)

    print(f"Weather plots saved to: {OUTPUT_DIR}")

def main():
    """
    Main function to load cleaned data and generate weather-related visualizations.
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

    create_weather_plots(bike_df, weather_df)

if __name__ == "__main__":
    main()