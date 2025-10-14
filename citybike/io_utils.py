import pandas as pd

def load_csv(path, **kwargs):
    """
    Load a CSV file with basic error handling.
    
    Args:
        path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded DataFrame.
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    try:
        return pd.read_csv(path, low_memory=False, **kwargs)
    except Exception as e:
        raise FileNotFoundError(f"File not found: {path}")
        
