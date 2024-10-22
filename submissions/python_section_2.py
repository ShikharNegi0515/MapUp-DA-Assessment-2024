import numpy as np
import pandas as pd
from datetime import time



def calculate_distance_matrix(df) -> pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame): The input dataframe containing columns 'id_start', 'id_end', 'distance'

    Returns:
        pandas.DataFrame: Distance matrix where each cell represents the distance between IDs
    """
    ids = pd.concat([df['id_start'], df['id_end']]).unique()
    
    distance_matrix = pd.DataFrame(np.inf, index=ids, columns=ids)
    
    np.fill_diagonal(distance_matrix.values, 0)
    
    for _, row in df.iterrows():
        start = row['id_start']
        end = row['id_end']
        distance = row['distance']
        
        distance_matrix.at[start, end] = distance
        distance_matrix.at[end, start] = distance 
    for k in ids:
        for i in ids:
            for j in ids:
                distance_matrix.at[i, j] = min(
                    distance_matrix.at[i, j], 
                    distance_matrix.at[i, k] + distance_matrix.at[k, j]
                )

    return distance_matrix



def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unrolls a distance matrix into a DataFrame with columns 'id_start', 'id_end', and 'distance'.

    Args:
        df (pandas.DataFrame): The input distance matrix DataFrame.

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    results = []
    ids = df.index.tolist()
    
    for id_start in ids:
        for id_end in ids:
            if id_start != id_end: 
                distance = df.at[id_start, id_end]
                results.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'distance': distance
                })
    unrolled_df = pd.DataFrame(results)
    
    return unrolled_df



def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> pd.DataFrame:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame): DataFrame with columns 'id_start', 'id_end', and 'distance'.
        reference_id (int): The reference id_start value.

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    reference_distances = df[df['id_start'] == reference_id]

    if reference_distances.empty:
        return pd.DataFrame(columns=['id_start', 'average_distance'])  

    average_distance_reference = reference_distances['distance'].mean()

    lower_threshold = average_distance_reference * 0.9
    upper_threshold = average_distance_reference * 1.1

    average_distances = df.groupby('id_start', as_index=False)['distance'].mean()
    average_distances.columns = ['id_start', 'average_distance']


    filtered_ids = average_distances[(average_distances['average_distance'] >= lower_threshold) & 
                                     (average_distances['average_distance'] <= upper_threshold)]

    return filtered_ids.sort_values(by='id_start')  # Return sorted DataFrame



def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing columns 'id_start', 'id_end', and 'distance'.

    Returns:
        pandas.DataFrame: DataFrame with toll rates for different vehicle types, excluding the 'distance' column.
    """
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    for vehicle, rate in rates.items():
        df[vehicle] = df['distance'] * rate
    
    
    return df



def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame): DataFrame containing vehicle toll rates.

    Returns:
        pandas.DataFrame: DataFrame with additional columns for start_day, start_time, end_day, and end_time,
                          and modified toll rates based on time intervals.
    """
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekends = ['Saturday', 'Sunday']

    new_rows = []
    for _, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance'] 

        for day in weekdays:
            time_intervals = [
                (time(0, 0), time(10, 0), 0.8),  # 00:00 to 10:00
                (time(10, 0), time(18, 0), 1.2), # 10:00 to 18:00
                (time(18, 0), time(23, 59, 59), 0.8)  # 18:00 to 23:59:59
            ]

            for start_time, end_time, discount in time_intervals:
                new_row = {
                    'id_start': id_start,
                    'id_end': id_end,
                    'start_day': day,
                    'start_time': start_time,
                    'end_day': day,
                    'end_time': end_time,
                    'moto': row['moto'] * discount,  # Adjust the toll rate for moto
                    'car': row['car'] * discount,      # Adjust the toll rate for car
                    'rv': row['rv'] * discount,        # Adjust the toll rate for rv
                    'bus': row['bus'] * discount,      # Adjust the toll rate for bus
                    'truck': row['truck'] * discount    # Adjust the toll rate for truck
                }
                new_rows.append(new_row)

        for day in weekends:

            start_time = time(0, 0)  # Starting from 00:00
            end_time = time(23, 59, 59)  # End at 23:59:59
            new_row = {
                'id_start': id_start,
                'id_end': id_end,
                'start_day': day,
                'start_time': start_time,
                'end_day': day,
                'end_time': end_time,
                'moto': row['moto'] * 0.7,  # Adjust the toll rate for moto
                'car': row['car'] * 0.7,      # Adjust the toll rate for car
                'rv': row['rv'] * 0.7,        # Adjust the toll rate for rv
                'bus': row['bus'] * 0.7,      # Adjust the toll rate for bus
                'truck': row['truck'] * 0.7    # Adjust the toll rate for truck
            }
            new_rows.append(new_row)

    result_df = pd.DataFrame(new_rows)

    return result_df



#  ---------------------------------------------------------- OPERATIONS ----------------------------------------------------



file_path = 'C:/Users/shikhar negi/Desktop/Udemy Python/Assignment/Mapup Assignment/MapUp-DA-Assessment-2024/datasets/dataset-2.csv'
data = pd.read_csv(file_path)
# print(data)


# Q9 - calculate_distance_matrix
distance_matrix_df = calculate_distance_matrix(data)
# print(distance_matrix_df)


# Q10 - unroll_distance_matrix
unrolled_df = unroll_distance_matrix(distance_matrix_df)
# print(unrolled_df)


# Q11 - find_ids_within_ten_percentage_threshold
reference_id = 1001400  # Example reference id
find_df = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)
# print(find_df)


# Q12 - calculate_toll_rate

toll_rate_df = calculate_toll_rate(unrolled_df)
# print(toll_rate_df)


# Q13 - calculate_time_based_toll_rates
time_df = calculate_time_based_toll_rates(toll_rate_df)
# print(time_df)