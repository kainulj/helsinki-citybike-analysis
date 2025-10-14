"""
Temporal analysis and visualization of Helsinki city bike usage.

This script explores how bike ride departures vary over different timeframes,
including yearly, monthly, weekly, and hourly patterns. It produces line and bar
plots to highlight seasonal and temporal usage trends.

All figures are saved to the 'figures/time' directory.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from citybike.io_utils import load_csv

# Style settings
sns.set_theme(context='paper',style='white', palette='Dark2')
plt.rcParams["figure.figsize"] = (8, 5)

# Paths
BIKE_PATH = Path("data/clean/bike_rides_cleaned.csv")
OUTPUT_DIR = Path("figures/temporal")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_yearly_usage(df):
    """
    Plot and save yearly bike usage as a bar chart.
    Args:
        df (pd.DataFrame): Dataframe containing ride data.
    """
    yearly_usage = df.groupby([df.departure.dt.year]).size()
    plt.figure()
    sns.barplot(yearly_usage)
    plt.title('Yearly Bike Usage')
    plt.xlabel('Year')
    plt.ylabel('Number of Departures')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'yearly_usage.png')
    plt.close()

def plot_monthly_usage(daily_usage):
    """
    Plot and save average daily usage by month as a line chart.
    Args:
        df (pd.DataFrame): Dataframe containing daily usage.
    """
    # Calculate average daily departures by month
    avg_daily_by_month = daily_usage.groupby("month")["departures"].mean().reset_index()

    # Plot average daily bike usage
    plt.figure()
    sns.lineplot(avg_daily_by_month, x='month', y='departures', marker='o')
    plt.title('Average Daily Bike Usage by Month (All Years Combined)')
    plt.xlabel('Month')
    plt.ylabel('Average Number of Departures')
    plt.xticks(ticks=np.arange(4,11), labels=['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct'], rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'monthly_usage_total.png')
    plt.close()

def plot_monthly_usage_by_year(daily_usage):
    """
    Plot and save average daily usage by year and month as a line chart.
    Args:
        daily_usage (pd.DataFrame): Dataframe containing daily usage.
    """
    # Calculate average daily departures by year and month
    avg_daily_by_year_month = daily_usage.groupby(['year', 'month'])["departures"].mean().reset_index()

    palette = sns.color_palette("Dark2", n_colors=len(daily_usage['year'].unique()))

    # Plot average daily bike usage by year
    plt.figure()
    sns.lineplot(data=avg_daily_by_year_month, x='month', y='departures', hue='year', marker='o', palette=palette)
    plt.title('Average Daily Bike Usage by Year and Month')
    plt.xlabel('Month')
    plt.ylabel('Average Number of Departures')
    plt.xticks(ticks=np.arange(4,11), labels=['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct'], rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'monthly_usage.png')
    plt.close()

def plot_weekday_usage(daily_usage):
    """
    Plot and save average daily bike usage by weekday as a bar chart.
    Args:
        daily_usage (pd.DataFrame): Dataframe containing daily usage.
    """
    # Calculate average daily departures by weekday
    avg_weekday_usage = daily_usage.groupby("weekday")["departures"].mean()

    # Plot average daily bike usage by weekday
    plt.figure()
    sns.barplot(avg_weekday_usage)
    plt.title('Average Daily Bike Usage by Weekday')
    plt.xlabel('Weekday')
    plt.ylabel('Average Number of Departures')
    plt.xticks(ticks=np.arange(0, 7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'weekday_usage.png')
    plt.close()

def plot_hourly_usage(hourly_usage):
    """
    Plot and save average hourly bike usage as a line chart.
    Args:
        hourly_usage (pd.DataFrame): Dataframe containing hourly departures.
    """
    hourly_usage['hour'] = hourly_usage.index.hour

    # Calculate average hourly usage
    avg_hourly_usage = hourly_usage.groupby('hour')['departures'].mean()

    plt.figure()
    sns.lineplot(avg_hourly_usage, marker='o')
    plt.title('Average Daily Bike Usage Patterns')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Number of Departures')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hourly_usage_total.png')
    plt.close()

def plot_hourly_usage_by_year(hourly_usage):
    """
    Plot and save average hourly bike usage by year as a line chart.
    Args:
        hourly_usage (pd.DataFrame): Dataframe containing ride data.
    """
    hourly_usage['hour'] = hourly_usage.index.hour
    hourly_usage['year'] = hourly_usage.index.year

    # Calculate average hourly usage by year
    avg_hourly_usage = hourly_usage.groupby(['year', 'hour'])['departures'].mean().reset_index()

    palette = sns.color_palette("Dark2", n_colors=len(hourly_usage['year'].unique()))

    plt.figure()
    sns.lineplot(data=avg_hourly_usage, x='hour', y='departures', hue='year', marker='o', palette=palette)
    plt.title('Average Hourly Bike Usage by Year')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Number of Departures')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hourly_usage.png')
    plt.close()

def plot_hourly_usage_by_daytype(hourly_usage):
    """
    Plot and save average hourly bike usage by weekday vs weekend as a line chart.
    Args:
        hourly_usage (pd.DataFrame): Dataframe containing ride data.
    """
    hourly_usage['is_weekend'] = hourly_usage.index.weekday > 4
    # Calculate average hourly usage for weekdays and weekends
    weekday_weekend_usage = hourly_usage.groupby(['is_weekend', 'hour'])['departures'].mean().reset_index()
    
    plt.figure()
    ax = sns.lineplot(weekday_weekend_usage, x='hour', y='departures', hue='is_weekend', marker='o')
    plt.title('Average Hourly Bike Usage: Weekday vs Weekend')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Number of Departures')
    handles, labels = ax.get_legend_handles_labels()
    new_labels = ['Weekdays', 'Weekends']
    plt.legend(handles, new_labels, title='Day Type')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hourly_usage_weekday_weekend.png')
    plt.close()

def create_temporal_plots(bike_df):
    """
    Creates time-based usage plots.
    Args:
        bike_df (pd.DataFrame): Dataframe containing ride data.
    """

    # Calculate hourly departures
    bike_df['time'] = bike_df['departure'].dt.floor('h')
    hourly_usage = bike_df.groupby(['time']).size().reset_index(name='departures')
    
    season_hours = pd.date_range(start=bike_df['time'].min(), 
                        end=bike_df['time'].max(), 
                        freq='h')
    season_hours = season_hours[(season_hours.month >= 4) & (season_hours.month <= 10)]
    hourly_usage = (hourly_usage.set_index(['time'])).reindex(season_hours, fill_value=0)

    hourly_usage['date'] = hourly_usage.index.date

    # Calculate total usage for each day
    daily_usage = hourly_usage.groupby('date').sum().reset_index()

    daily_usage["month"] = pd.to_datetime(daily_usage["date"]).dt.month
    daily_usage["year"] = pd.to_datetime(daily_usage["date"]).dt.year
    daily_usage['weekday'] = pd.to_datetime(daily_usage["date"]).dt.weekday

    # Create the plots
    plot_yearly_usage(bike_df)
    plot_monthly_usage(daily_usage)
    plot_monthly_usage_by_year(daily_usage)
    plot_weekday_usage(daily_usage)
    plot_hourly_usage(hourly_usage)
    plot_hourly_usage_by_year(hourly_usage)
    plot_hourly_usage_by_daytype(hourly_usage)
    
    print(f"Temporal plots saved to: {OUTPUT_DIR}")

def main():
    """
    Main function to load cleaned data and generate time-based usage plots.
    """
    # Load the data
    dtypes = {'departure_id': str, 'departure_name': str, 
            'return_id': str, 'return_name': str}
    
    try:
        bike_df = load_csv(BIKE_PATH, dtype=dtypes, parse_dates=['departure', 'return'])
    except FileNotFoundError as e:
        print(e)
        return

    create_temporal_plots(bike_df)
    

if __name__ == "__main__":
    main()