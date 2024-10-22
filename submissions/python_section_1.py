from typing import Dict, List
from datetime import datetime, timedelta
import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    for i in range(0, len(lst), n):
        group = lst[i:i+n]
        reversed_group = []
        for j in range(len(group)-1, -1, -1):
            reversed_group.append(group[j])
        result.extend(reversed_group)
    return result


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    result = {}
    for s in lst:
        length = len(s)
        if length not in result:
            result[length] = []
        result[length].append(s)
    return dict(sorted(result.items()))

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    def flatten_helper(d, parent_key=''):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_helper(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    return flatten_helper(nested_dict)

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    Args:
        nums (List[int]): List of integers that may contain duplicates.
    
    Returns:
        List[List[int]]: A list of unique permutations.
    """
    from itertools import permutations
    return [list(p) for p in set(permutations(nums))]


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Args:
        text (str): A string containing the dates in various formats.

    Returns:
        List[str]: A list of valid dates found in the string.
    """
    import re
    date_pattern = r'(\d{2}-\d{2}-\d{4})|(\d{2}/\d{2}/\d{4})|(\d{4}\.\d{2}\.\d{2})'
    matches = re.findall(date_pattern, text)
    dates = ["".join(match) for match in matches]
    return dates

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    import polyline
    from math import radians, sin, cos, sqrt, atan2
    
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000  # Earth radius in meters
        dlat = radians(lat2 - lat1)
        dlon = radians(lon1 - lon2)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c
    
    coords = polyline.decode(polyline_str)
    df = pd.DataFrame(coords, columns=['latitude', 'longitude'])
    df['distance'] = 0
    for i in range(1, len(df)):
        df.loc[i, 'distance'] = haversine(df.loc[i-1, 'latitude'], df.loc[i-1, 'longitude'], df.loc[i, 'latitude'], df.loc[i, 'longitude'])
    return df


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
        matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
        List[List[int]]: A new 2D list representing the transformed matrix.
    """
    import numpy as np
    n = len(matrix)
    matrix = np.array(matrix)
    rotated = np.rot90(matrix, -1)  # Rotate 90 degrees clockwise
    transformed = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(n):
            row_sum = np.sum(rotated[i, :]) - rotated[i, j]
            col_sum = np.sum(rotated[:, j]) - rotated[i, j]
            transformed[i, j] = row_sum + col_sum
    return transformed.tolist()


def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7-day period.

    Args:
        df (pandas.DataFrame): Input DataFrame containing the dataset.

    Returns:
        pd.Series: A boolean series indicating if each (id, id_2) pair has incorrect timestamps.
    """
    week_days = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }
    
    start_date = datetime(2023, 1, 1)  
    day_offset = {day: start_date + timedelta(days=index) for day, index in week_days.items()}

    df['start_datetime'] = df.apply(
        lambda x: day_offset[x['startDay']] + pd.to_timedelta(x['startTime']),
        axis=1
    )
    df['end_datetime'] = df.apply(
        lambda x: day_offset[x['endDay']] + pd.to_timedelta(x['endTime']),
        axis=1
    )

    grouped = df.groupby(['id', 'id_2'])

    def check_completeness(group):
        unique_days = group['start_datetime'].dt.day_name().unique()
        all_days_covered = set(unique_days) == {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'}
        
        start_of_period = group['start_datetime'].min().normalize() 
        end_of_period = group['end_datetime'].max().normalize() + pd.Timedelta(days=1)  
        full_day_covered = (end_of_period - start_of_period) >= pd.Timedelta(days=7)

        return not (all_days_covered and full_day_covered)

    completeness_series = grouped.apply(check_completeness)

    return completeness_series


# file_path = 'C:/Users/shikhar negi/Desktop/Udemy Python/Assignment/Mapup Assignment/MapUp-DA-Assessment-2024/datasets/dataset-1.csv'
# data = pd.read_csv(file_path)
# result = time_check(data)


