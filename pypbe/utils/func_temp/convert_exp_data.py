# -*- coding: utf-8 -*-
"""
Convert experimental data into data that can be directly input into opt
"""
import os
import re
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from datetime import timedelta

def load_excel_data(file_path):
    # Load the Excel file
    excel_data = pd.ExcelFile(file_path)
    
    # Initialize a dictionary to hold data from all sheets
    all_data = {}
    
    for sheet_name in excel_data.sheet_names:
        # Load data from each sheet
        sheet_data = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Find all unique time labels
        time_labels = sheet_data.iloc[1].dropna().unique()
        
        # Initialize a dictionary to hold data for this sheet
        sheet_dict = {}
        
        for label in time_labels:
            # Get the starting column index for the current label
            start_col = sheet_data.columns.get_loc(sheet_data.columns[sheet_data.iloc[1] == label][0])
            
            # Initialize lists to hold the data
            x = []
            Q_x = []
            xm = []
            q_lnx = []
            
            # Determine the number of measurements by counting occurrences of the label
            measurement_count = (sheet_data.iloc[1] == label).sum()
            
            # Replace NaN values with 0 in the entire sheet
            sheet_data = sheet_data.fillna(0)
            
            for i in range(measurement_count):
                col_offset = i * 5
                x_data = sheet_data.iloc[5:len_data+5, start_col + col_offset].values
                Q_x_data = sheet_data.iloc[5:len_data+5, start_col + col_offset + 1].values
                xm_data = sheet_data.iloc[5:len_data+5, start_col + col_offset + 2].values
                q_lnx_data = sheet_data.iloc[5:len_data+5, start_col + col_offset + 3].values
                
                # Replace NaN with 0
                x.append(np.nan_to_num(x_data, nan=0))
                Q_x.append(np.nan_to_num(Q_x_data, nan=0))
                xm.append(np.nan_to_num(xm_data, nan=0))
                q_lnx.append(np.nan_to_num(q_lnx_data, nan=0))
            # Store the data in the dictionary
            sheet_dict[label] = {
                'x': x,
                'Q(x)': Q_x,
                'xm': xm,
                'q(lnx)': q_lnx
            }
        
        # Store the sheet data in the main dictionary
        all_data[sheet_name] = sheet_dict
        
    merged_data = {}
    for sheet_data in all_data.values():
        merged_data.update(sheet_data)
    
    return merged_data, measurement_count

def extract_minutes(label):
    match = re.search(r'(\d+) min', label)
    return int(match.group(1)) if match else 0

def interpolate_data(original_data, original_coords, new_coords):
    interpolated_data = np.zeros((original_data.shape[0], original_data.shape[1], len(new_coords)))
    for i in range(original_data.shape[0]):
        for j in range(original_data.shape[1]):
            f = interp1d(original_coords[i, j, :], original_data[i, j, :], bounds_error=False, fill_value="extrapolate")
            interpolated_data[i, j, :] = f(new_coords)
    interpolated_data[np.where(interpolated_data<0)] = 0.0
    return interpolated_data

def generate_nonuniform_coords(min_val, max_val, num_points):
    log_min = np.log10(min_val + 1e-10)  # Avoid log(0)
    log_max = np.log10(max_val)
    log_coords = np.linspace(log_min, log_max, num_points)
    return np.power(10, log_coords)
def process_data(merged_data, measurement_count):
    # Sort the time labels to ensure chronological order
    sorted_time_labels = sorted(merged_data.keys(), key=extract_minutes)

    # Create 3D arrays for each type of measurement
    Q_x_array = np.zeros((measurement_count, len(sorted_time_labels), len_data))
    q_lnx_array = np.zeros((measurement_count, len(sorted_time_labels), len_data))
    q_x_array = np.zeros((measurement_count, len(sorted_time_labels), len_data))
    x_array = np.zeros((measurement_count, len(sorted_time_labels), len_data))
    xm_array = np.zeros((measurement_count, len(sorted_time_labels), len_data))
    
    for time_label in sorted_time_labels:
        measurements = merged_data[time_label]
        Q_x = measurements['Q(x)']
        q_lnx = measurements['q(lnx)']
        x = measurements['x']
        xm = measurements['xm']

        for i in range(measurement_count):
            Q_x_array[i, sorted_time_labels.index(time_label), :] = Q_x[i]
            q_lnx_array[i, sorted_time_labels.index(time_label), :] = q_lnx[i]
            x_array[i, sorted_time_labels.index(time_label), :] = x[i]
            xm_array[i, sorted_time_labels.index(time_label), :] = xm[i]
    q_x_array[:,:,1:] = q_lnx_array[:,:,1:] / xm_array[:,:,1:]
    # Find the global min and max values for x and xm
    x_min, x_max = np.min(x_array), np.max(x_array)
    xm_min, xm_max = np.min(xm_array[np.nonzero(xm_array)]), np.max(xm_array)
    
    # Generate new coordinates
    new_xm_coords = np.zeros(200)
    new_x_coords = generate_nonuniform_coords(x_min, x_max, 200)
    new_xm_coords[1:] = generate_nonuniform_coords(xm_min, xm_max, 199)
    Q_x_int_array = interpolate_data(Q_x_array, x_array, new_x_coords)
    q_x_int_array = interpolate_data(q_x_array, xm_array, new_xm_coords)
    ## Theoretically, q_x also needs to be normalized, 
    ## but it needs to be integrated, which will cause a larger error!!!
    for i in range(Q_x_int_array.shape[0]):
        for j in range(Q_x_int_array.shape[1]):
            scale = Q_x_int_array[i,j,199]
            Q_x_int_array[i,j,:] /= scale
    
    return Q_x_int_array, q_x_int_array, new_x_coords, new_xm_coords, sorted_time_labels


def save_interpolated_data(Q_x_int_array, q_x_int_array, x_arrays, xm_arrays, sorted_time_labels, measurement_count):
    # Extract minutes from sorted_time_labels and convert to '%H:%M:%S' format
    sorted_minutes = [extract_minutes(label) for label in sorted_time_labels]
    formatted_time_labels = [str(timedelta(minutes=minutes)) for minutes in sorted_minutes]
    
    # Create a list to store dataframes
    Q_x_dfs = []
    q_x_dfs = []

    for i in range(measurement_count):
        # Create dataframe for Q_x_int_array
        Q_x_df = pd.DataFrame(Q_x_int_array[i].T, index=x_arrays, columns=formatted_time_labels)
        Q_x_df.index.name = 'Circular Equivalent Diameter'
        Q_x_dfs.append(Q_x_df)

        # Create dataframe for q_x_int_array
        q_x_df = pd.DataFrame(q_x_int_array[i].T, index=xm_arrays, columns=formatted_time_labels)
        q_x_df.index.name = 'Circular Equivalent Diameter'
        q_x_dfs.append(q_x_df)
        save_path = os.path.join("Batch_Messung", f"Batchversuch_600rpm_1200rpm_{i}.xlsx")
        with pd.ExcelWriter(save_path) as writer:
            Q_x_df.to_excel(writer, sheet_name='Q_x')
            q_x_df.to_excel(writer, sheet_name='q_x')
    return Q_x_dfs, q_x_dfs
            
if __name__ == '__main__':
    len_data = 200 
    # Usage
    file_path = os.path.join("Batch_Messung", "462DB100.xlsx")
    data, measurement_count = load_excel_data(file_path)
    Q_x_arrays, q_x_array, x_arrays, xm_arrays, sorted_time_labels = process_data(data, measurement_count)
    Q_x_dfs, q_x_dfs = save_interpolated_data(Q_x_arrays, q_x_array, x_arrays, xm_arrays, sorted_time_labels,measurement_count)
    # Now `data` contains all the extracted information


