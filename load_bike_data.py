import requests, zipfile, io
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Download data script")

parser.add_argument('--start-year', type=int, required=True, help='Start year of data range')
parser.add_argument('--end-year', type=int, required=True, help='End year of data range')
parser.add_argument('--output', type=str, default='data.csv', help='Output file name')

dtypes = {'Departure': str, 'Return': str, 'Departure_id': str, 'Departure_name': str, 
            'Return_id': str, 'Return_name': str}
columns = ['Departure', 'Return', 'Departure_id', 'Departure_name', 'Return_id', 'Return_name', 'Distance', 'Duration']
dfs = []

args = parser.parse_args()

print(f"Downloading data from {args.start_year} to {args.end_year}")

for year in range(args.start_year, args.end_year + 1):
    # Load the data
    r = requests.get(f'http://dev.hsl.fi/citybikes/od-trips-{year}/od-trips-{year}.zip')

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        # Loop trough the files in the zip archive
        for file_name in z.namelist():
            if file_name.endswith('.csv'):
                # Read CSV file to dataframe
                with z.open(file_name) as f:
                    df = pd.read_csv(f,names=columns, header=0, dtype=dtypes)
                    dfs.append(df)

# Combine the data into one notebook
bike_df = pd.concat(dfs, ignore_index=True)

# Change the departure and return colu,ms to date time and sort the dataframe by the deaprture time
bike_df['Departure'] = pd.to_datetime(bike_df['Departure'], errors='coerce')
bike_df['Return'] = pd.to_datetime(bike_df['Return'], errors='coerce')
bike_df.sort_values(by=['Departure'], inplace=True)

print(f"Saving to {args.output}")
bike_df.to_csv(args.output, index=False)