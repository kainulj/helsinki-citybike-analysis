"""
Distribution analysis of Helsinki city bike trip characteristics.

This script visualizes the distributions of trip durations, distances, and
ride counts to identify general usage patterns and outliers. It produces
histograms and density plots for key ride metrics.

All figures are saved to the 'figures/distributions' directory.
"""
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
OUTPUT_DIR = Path("figures/distributions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_histograms(df, cols):
    """
    Plot and save histograms for specified columns in the dataframe.
    Args:
        df (pd.DataFrame): Dataframe containing ride data.
        cols (list): List of column names to plot.
        path (str): Directory to save plots.
    """
    for col in cols:
        plt.figure()
        sns.histplot(df[col], bins=60)
        plt.title(f'Distribution of Ride {col.capitalize()}')
        plt.xlabel(col.capitalize())
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'hist_{col}.png')
        plt.close()

def plot_histograms_pricechange(df, cols, cutoff):
    """
    Plot histograms to visualize the effect of a pricing change on ride features.
    Args:
        df (pd.DataFrame): Dataframe containing ride data.
        cols (list): List of column names to plot.
        path (str): Directory to save plots.
        cutoff (int): Year of pricing change.
    """
    for col in cols:
        fig, ax = plt.subplots(1,2, figsize=(10, 5))
        fig.suptitle(f"The effect of the pricing change on the ride {col} on short trips (< 60 min)", fontweight='bold')
        sns.histplot(df[(df['duration'] < 60) & (df.year < cutoff)][col], bins=50, ax=ax[0])
        ax[0].set_title(f'Old princing (before {cutoff})')
        ax[0].set_xlabel(col.capitalize())

        sns.histplot(df[(df['duration'] < 60) & (df.year >= cutoff)][col], bins=50, ax=ax[1])
        ax[1].set_title(f'New pricing ({cutoff})')
        ax[1].set_xlabel(col.capitalize())
        ax[1].set_ylabel('')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'hist_{col}_pricechange.png')
        plt.close()

def create_distribution_plots(bike_df):
    """
    Creates distribution plots.
    Args:
        bike_df (pd.DataFrame): Dataframe containing ride data.
    """

    # Speed in km/h
    bike_df['speed'] = (bike_df['distance'] / 1000) / (bike_df['duration'] / 60) 

    bike_df['year'] = bike_df['departure'].dt.year

    # Create the histograms
    plot_histograms(bike_df, ['duration', 'distance', 'speed'])
    plot_histograms_pricechange(bike_df, ['duration'], 2024)

    print(f"Distribution plots saved to: {OUTPUT_DIR}")

def main():
    """
    Main function to load cleaned trip data and generate distribution plots.
    """
    
    dtypes = {'departure_id': str, 'departure_name': str, 
            'return_id': str, 'return_name': str}
    # Load the data
    try:
        bike_df = load_csv(BIKE_PATH, dtype=dtypes, parse_dates=['departure', 'return'])
    except FileNotFoundError as e:
        print(e)
        return
    
    create_distribution_plots(bike_df)

if __name__ == "__main__":
    main()