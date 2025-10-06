def _add_lags(group, column, freq, lags):
    """
    Adds lagged columns for a target feature.

    Parameters:
        group (pd.DataFrame): Data for a single station, indexed by time.
        column (str): Name of the target column to lag.
        freq (str): Frequency string (e.g., '1H') for time-based shifting.
        lags (list of int): List of lag periods to add.

    Returns:
        pd.DataFrame: DataFrame with new lagged columns added.
    """
    group = group.sort_index()
    for lag in lags:
        group[f'{column[:3]}_lag_{lag}'] = group[column].shift(lag, freq=freq)
    return group

def _add_rolling_means(group, column, freq, windows):
    """
    Adds rolling mean and standard deviation columns for a target feature.

    Parameters:
        group (pd.DataFrame): Data for a single station, indexed by time.
        column (str): Name of the target column.
        freq (str): Frequency string for time-based shifting.
        windows (list of int): List of window sizes for rolling calculations.

    Returns:
        pd.DataFrame: DataFrame with new rolling mean and std columns added.
    """
    group = group.sort_index()
    for window in windows:
        group[f'{column[:3]}_roll_mean_{window}'] = group[column].shift(1, freq=freq).rolling(window=window).mean()
        group[f'{column[:3]}_roll_std_{window}'] = group[column].shift(1, freq=freq).rolling(window=window).std()
    return group

def _same_hour_rolling_mean(group, column, freq, windows):
    """
    Adds rolling mean and std columns for the same hour of the day across previous days.

    Parameters:
        group (pd.DataFrame): Data for a single station, indexed by time.
        column (str): Name of the target column.
        freq (str): Frequency string for time-based shifting.
        windows (list of int): List of window sizes (in days) for rolling calculations.

    Returns:
        pd.DataFrame: DataFrame with new same-hour rolling mean and std columns added.
    """
    group = group.sort_index()
    dummy_group = group.copy()
    dummy_cols = []
    # Create temporary columns with values from the same hour in previous days
    for i in range(1, max(windows)+1):
        dummy_col = f'{column[:3]}_same_hour_roll_mean_{i}'
        dummy_group[dummy_col] = group[column].shift(i*24, freq=freq)
        dummy_cols.append(dummy_col)
    # Calculate rolling statistics based on the temporary columns
    for window in windows:
        window_cols = dummy_cols[:window]
        group[f'{column[:3]}_same_hour_mean_{window}d'] = dummy_group[window_cols].mean(axis=1)
        group[f'{column[:3]}_same_hour_std_{window}d'] = dummy_group[window_cols].std(axis=1)
    return group

def _shift_weather_cols(group, weather_cols, freq):
    """
    Shifts weather columns backward by one period to prevent data leakage.

    Parameters:
        group (pd.DataFrame): Data for a single station, indexed by time.
        weather_cols (list of str): List of weather column names to shift.
        freq (str): Frequency string for time-based shifting.

    Returns:
        pd.DataFrame: DataFrame with shifted weather columns.
    """
    group = group.sort_index()
    for col in weather_cols:
        group[col] = group[col].shift(1, freq=freq)
    return group

def add_features(df, target_col, lags, rolling_windows, same_hour_windows, weather_cols, freq):
    """
    Adds lag, rolling, temporal, and weather features to the DataFrame.

    This function:
      - Adds lag features for the target column.
      - Adds rolling mean and std features.
      - Adds rolling statistics for the same hour across previous days.
      - Adds time-based features (hour, weekday, month, year, is_weekend).
      - Shifts weather columns to avoid data leakage.
      - Adds a binary rain indicator.
      - Drops rows with missing values and sorts the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame with a datetime index or 'time' column.
        target_col (str): Name of the target column.
        lags (list of int): List of lag periods to add.
        rolling_windows (list of int): List of window sizes for rolling features.
        same_hour_windows (list of int): List of window sizes (in days) for same-hour rolling features.
        weather_cols (list of str): List of weather column names to shift.
        freq (str): Frequency string for time-based shifting.

    Returns:
        pd.DataFrame: DataFrame with new features added.
    """
    if df.index.name != 'time':
        df = df.set_index('time')

    # Add lag features for the target column
    df = df.groupby('station_id', group_keys=False, observed=True)[df.columns].apply(_add_lags, target_col, lags=lags, freq=freq)
    
    # Add rolling mean and std features
    df = df.groupby('station_id', group_keys=False, observed=True)[df.columns].apply(_add_rolling_means, target_col, windows=rolling_windows, freq=freq)
    
    # Add rolling statistics for the same hour across previous days
    df = df.groupby('station_id', group_keys=False, observed=True)[df.columns].apply(_same_hour_rolling_mean, target_col, windows=same_hour_windows, freq=freq)

    # Add time-based features (hour, weekday, etc.)
    df['hour'] = df.index.hour
    df[f'weekday'] = df.index.weekday
    df[f'month'] = df.index.month
    df[f'is_weekend'] = df.index.weekday.isin([5, 6]).astype(int)
    df[f'year'] = df.index.year

    df['station_id'] = df['station_id'].astype('category')

    # Shift weather columns to avoid data leakage
    df = df.groupby('station_id', group_keys=False, observed=True)[df.columns].apply(_shift_weather_cols, weather_cols, freq=freq)  
    
    # Create binary rain indicator
    df['rain'] = (df['precipitation'] > 0).astype(int)

    # Drop rows with missing values and sort
    df = df.dropna()
    df = df.sort_values(by=['time', 'station_id'])
    df = df.reset_index(drop=False)

    return df