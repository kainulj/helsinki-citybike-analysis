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

def handle_wind_speed_gaps(df):
    is_na = df['wind_speed'].isna().astype(int)
    groups = (~is_na.astype(bool)).cumsum()
    gap_lengths = is_na.groupby(groups).sum()

    # Interpolate short gaps linearly
    for grp, length in gap_lengths.items():
        if length > 0 and length <= 6:
            idx = df['wind_speed'][groups == grp].index
            df.loc[idx, 'wind_speed'] = df['wind_speed'].loc[idx].interpolate(method='linear')

    # Flag longer gaps and replace with -1
    df['ws_missing'] = df['wind_speed'].isna().astype(int)
    df['wind_speed'] = df['wind_speed'].fillna(-1)

    return df
