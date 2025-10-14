"""
Spatial analysis and visualization of Helsinki city bike stations.

This script examines the geographic distribution of bike stations and usage
patterns across the city. It generates static plots and interactive Folium maps
illustrating station density, station usage, and most traveled routes.

All figures and maps are saved to the 'figures/spatial' directory.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from pathlib import Path
from citybike.io_utils import load_csv

# Style settings
sns.set_theme(context='paper',style='white')
palette = sns.color_palette("Dark2")
sns.set_palette(palette)
plt.rcParams["figure.figsize"] = (8, 5)

# Paths
BIKE_PATH = Path("data/clean/bike_rides_cleaned.csv")
OUTPUT_DIR = Path("figures/spatial")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def aggregate_station_counts(df):
    """
    Aggregate total trips for each station by combining departures and returns.
    Args:
        df (pd.DataFrame): Dataframe containing ride data.
    Returns:
        pd.DataFrame: Dataframe with station trip counts and coordinates.
    """
    # Count departures for each station
    station_departure_counts = (
        df.groupby(['departure_name', 'departure_lat', 'departure_lon'])
        .size()
        .reset_index(name='departures')
        .rename(columns={'departure_name': 'station', 'departure_lat': 'lat', 'departure_lon': 'lon'})
    )

    # Count returns for each station
    station_return_counts = (
        df.groupby(['return_name', 'return_lat', 'return_lon'])
        .size()
        .reset_index(name='returns')
        .rename(columns={'return_name': 'station', 'return_lat': 'lat', 'return_lon': 'lon'})
    )

    # Merge departure and return counts
    station_counts = pd.merge(
        station_departure_counts,
        station_return_counts,
        on=['station', 'lat', 'lon'],
        how='outer'
    ).fillna(0)

    # Calculate total trips per station
    station_counts['total_trips'] = station_counts['departures'] + station_counts['returns']
    station_counts = station_counts.sort_values(by='total_trips', ascending=False).reset_index(drop=True)

    return station_counts

def plot_top_stations(df):
    """
    Plot and save a bar chart of the top 10 stations by total trips.
    Args:
        df (pd.DataFrame): Dataframe with station trip counts.
    """
    plt.figure()
    sns.barplot(data=df.head(10), x='station', y='total_trips')
    plt.title('Top 10 Stations by Total Trips')
    plt.xlabel('Station')
    plt.ylabel('Number of Total Trips')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'top_stations.png')
    plt.close()

def plot_top_station_pairs(df):
    """
    Plot and save a bar chart of the top 10 most common station pairs.
    Args:
        df (pd.DataFrame): Dataframe containing ride data.
    """
    df['station_pair'] = df['departure_name'] + " → " + df['return_name']
    # Count trips for each station pair
    pair_counts = df['station_pair'].value_counts().head(10)

    plt.figure()
    sns.barplot(
        x=pair_counts.values,
        y=pair_counts.index,
    )
    plt.title('Top 10 Most Common Station Pairs')
    plt.xlabel('Number of Trips')
    plt.ylabel('Station Pair')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'top_station_pairs.png')
    plt.close()

def plot_station_map(df):
    """
    Plot and save a map of stations with ride counts visualized by marker size/color.
    Args:
        df (pd.DataFrame): Dataframe with station coordinates and trip counts.
    """
    # City center coordinates
    city_center_lat = 60.1699
    city_center_lon = 24.9384

    # Calculate quartile bins for ride counts
    df['usage_quartile'] = pd.qcut(df['total_trips'], 4, labels=False)
    quartile_colors = {0: '#ffffb2', 1: '#fecc5c', 2: '#fd8d3c', 3: '#e31a1c'}
    df['color'] = df['usage_quartile'].map(quartile_colors)

    # Create a folium map centered on the Helsinki city center
    m = folium.Map(location=[city_center_lat, city_center_lon], zoom_start=12)

    # Add a marker indicating the city center
    folium.Marker(
        location=[city_center_lat, city_center_lon],
        popup="Helsinki City Center"
    ).add_to(m)

    # Plot departure stations, with color indicating the usage quartile
    for _, row in df[df['lat'].notna()].iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=5,
            color='black',
            weight=1,
            fill=True,
            fill_color=row['color'],
            fill_opacity=0.6,
            popup=f"{row['station']}: {row['total_trips']} trips"
        ).add_to(m)

    legend_html = """
    <div style="position: fixed;
                bottom: 30px; left: 30px; width: 160px; height: 140px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white; padding: 10px;">
        <strong>Station Usage Quartiles</strong><br>
        <div style="background-color: #ffffb2; width: 10px; height: 10px; display: inline-block;"></div> Low Usage<br>
        <div style="background-color: #fecc5c; width: 10px; height: 10px; display: inline-block;"></div> Medium Usage<br>
        <div style="background-color: #fd8d3c; width: 10px; height: 10px; display: inline-block;"></div> High Usage<br>
        <div style="background-color: #e31a1c; width: 10px; height: 10px; display: inline-block;"></div> Very High Usage
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(OUTPUT_DIR / 'station_map.html')

def plot_most_popular_routes(df, top_n):
    """
    Plot and save a map visualizing the most popular routes between stations.
    Args:
        df (pd.DataFrame): Dataframe containing ride data and station coordinates.
    """
    city_center_lat = 60.1699
    city_center_lon = 24.9384

    # Create a bidirectional route key so each station pair shares the same identifier
    stations_min = df[['departure_name', 'return_name']].min(axis=1)
    stations_max = df[['departure_name', 'return_name']].max(axis=1)
    df['pair_key'] = stations_min + " ↔ " + stations_max

    # Sort the routes by the number of trips
    pair_counts = df.groupby('pair_key').size().reset_index(name='trip_count')
    pair_counts = pair_counts.sort_values(by='trip_count', ascending=False).reset_index(drop=True)

    stations_coords = df[['departure_name', 'departure_lat', 'departure_lon']].drop_duplicates().set_index('departure_name')

    # Create a map centered on the Helsinki city center
    m = folium.Map(location=[city_center_lat, city_center_lon], zoom_start=12)

    # Plot the top n routes on the map
    for _, row in pair_counts.head(top_n).iterrows():
        stations = row['pair_key'].split(" ↔ ")

        start_coords = stations_coords.loc[stations[0], ['departure_lat', 'departure_lon']].values
        end_coords = stations_coords.loc[stations[1], ['departure_lat', 'departure_lon']].values
        
        # Add Circle markers to the stations
        folium.CircleMarker(
            location=start_coords,
            radius=3,
            color='red',
            fill=True,
            fill_opacity=0.6,
            popup=f"{stations[0]}"
        ).add_to(m)

        folium.CircleMarker(
            location=end_coords,
            radius=3,
            color='red',
            fill=True,
            fill_opacity=0.6,
            popup=f"{stations[1]}"
        ).add_to(m)

        # Add line between the stations
        folium.PolyLine(
            locations=[start_coords, end_coords],
            color='blue',
            weight=3,
            opacity=0.6,
            popup=f"{row['pair_key']}: {row['trip_count']} trips"
        ).add_to(m)
    m.save(OUTPUT_DIR /'routes_map.html')

def create_spatial_plots(bike_df):
    """
    Creates spatial usage visualizations.
    Args:
        bike_df (pd.DataFrame): Dataframe containing ride data and station coordinates.
    """

    station_counts = aggregate_station_counts(bike_df)

    plot_top_stations(station_counts)
    plot_top_station_pairs(bike_df)
    plot_station_map(station_counts)
    plot_most_popular_routes(bike_df, 20)

    print(f"Spatial plots and maps saved to: {OUTPUT_DIR}")

def main():
    """
    Main function to load station data and create spatial usage visualizations.
    """
    # Load the data
    dtypes = {'departure_id': str, 'departure_name': str, 
            'return_id': str, 'return_name': str}
    
    try:
        bike_df = load_csv(BIKE_PATH, dtype=dtypes, parse_dates=['departure', 'return'])
    except FileNotFoundError as e:
        print(e)
        return

    create_spatial_plots(bike_df)

if __name__ == "__main__":
    main()