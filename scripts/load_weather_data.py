"""
This script downloads weather data from the Finnish Meteorolocigal Institute (FIM) API
and outputs a CSV file with the following columns:

- time: Observation time (datetime)
- air temperature: air temperature in °C (float)
- wind speed; wind speed in m/s (float)
- precipitation amount: precipitation in mm (float)
"""

import argparse
import os
import pandas as pd
from fmiopendata.wfs import download_stored_query
from datetime import datetime, timedelta
import pytz

def fetch_fmi_data(station, params,start_time, end_time, local_time, utc):
    # Conver the times to UTC
    start = local_time.localize(start_time).astimezone(utc).replace(tzinfo=None)
    end = local_time.localize(end_time).astimezone(utc).replace(tzinfo=None)

    # FMI API has limit of 7 days of data per request
    step = timedelta(days=7)
    results = []
    current = start
    while current < end:
        next_time = min(current + step, end)

        obs = download_stored_query(
            "fmi::observations::weather::multipointcoverage",
            args=[
                f"place={station}",
                f"parameters={params}",
                f"starttime={current.isoformat()}Z",
                f"endtime={next_time.isoformat()}Z",
                "timestep=60",
                "timeseries=True"
            ]
        )

        data = next(iter(obs.data.values()))
        df = pd.DataFrame({
            'time': data['times'],
            **{k: v['values'] for k, v in data.items() if k != 'times'}
        })
        results.append(df)

        # Offset chuck borders by one hour to avoid duplicated rows
        current = next_time + timedelta(hours=1)
    return pd.concat(results)

def main(start_year, end_year, output):
    local_time = pytz.timezone("Europe/Helsinki")
    utc = pytz.UTC

    # Get temperature, windspeed and precipitation values
    params = "t2m,ws_10min,r_1h"

    all_years = []
    for year in range(start_year, end_year + 1):
        start = datetime(year, 4, 1, 0, 0)
        end = datetime(year, 10, 31, 23, 59)
        df = fetch_fmi_data("Helsinki", params, start, end, local_time, utc)
        all_years.append(df)
    full_df = pd.concat(all_years)

    # Convert the time to Helsinki time
    full_df['time'] = full_df['time'].dt.tz_localize(utc).dt.tz_convert(local_time).dt.tz_localize(None)

    full_df.reset_index(inplace=True, drop=True)

    full_df.rename(columns={
        'Air temperature': 'temperature',
        'Wind speed': 'wind_speed',
        'Precipitation amount': 'precipitation'
    }, inplace=True)

    full_df.to_csv(output, index=False)
    print(f"Saved weather data to {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch station locations script")

    parser.add_argument('--start-year', type=int, default=2020, help='Start year of data range')
    parser.add_argument('--end-year', type=int, default=2024, help='End year of data range')
    parser.add_argument('--output', type=str, default='data/raw/weather.csv', help='Output CSV file path')

    args = parser.parse_args()

    # Create the output folder if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    main(args.start_year, args.end_year, args.output)