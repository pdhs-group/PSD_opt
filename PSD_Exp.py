# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:59:35 2023

Read or interpolate experimental data based on the time step of the simulated data

@author: px2030
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import interp1d

class write_read_exp():
    
    def __init__(self, path_exp_data = None, read = False):
        
        if read:
            # read the original data
            self.exp_data = pd.read_excel(path_exp_data)
            self.exp_data = self.exp_data.set_index('Circular Equivalent Diameter')
                    
            # Parse time in Format %H:%M:%S and convert to minutes
            self.exp_data.columns = [self.convert_time_to_seconds(col) for col in self.exp_data.columns]
            
            # Merge data from duplicate time columns
            self.merge_duplicate_time_columns()

        
    def convert_time_to_seconds(self, time_str):
        # Parse the time string into a datetime object and then convert it to second
        try:
            # First try the format without milliseconds
            time_obj = datetime.strptime(time_str, '%H:%M:%S')
        except ValueError:
            try:
                time_obj = datetime.strptime(time_str, '%H:%M:%S.%f')
            except ValueError:
                raise ValueError(f"Time string format not recognized: {time_str}")
                
        total_second = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second  
        # print(f"Time string: {time_str}, Total second: {total_second}")
        
        return total_second
    
    def convert_seconds_to_time(t_vec):
        # Parse the datetime into time string
        formatted_times = [str(timedelta(seconds=t)).split(".")[0] + '.' + str(int(t % 1 * 1000000)).zfill(6) for t in t_vec]
        
        return formatted_times

    def merge_duplicate_time_columns(self):

        unique_times = []
        duplicate_columns = []

        # Identify repeating time columns
        for col in self.exp_data.columns:
            if col not in unique_times:
                unique_times.append(col)
            else:
                duplicate_columns.append(col)

        # For each repeated time point, merge the data 
        for time in duplicate_columns:
            duplicate_data = self.exp_data.loc[:, self.exp_data.columns == time]
            mean_data = duplicate_data.mean(axis=1)
            self.exp_data = self.exp_data.drop(columns=duplicate_data.columns)
            self.exp_data[time] = mean_data

        # Rearrange the columns so they are in chronological order 
        self.exp_data = self.exp_data.reindex(sorted(self.exp_data.columns), axis=1)
        
    def get_exp_data(self, t_exp):
        # if t_exp is in the experiment data
        if t_exp in self.exp_data.columns:
            return self.exp_data[t_exp]
        else:
            # use interpolation to find experiment data
            time_points = self.exp_data.columns[:]  
            
            # make sure that t_exp not out of the range
            if t_exp < min(time_points) or t_exp > max(time_points):
                raise ValueError("The experimental time is out of the range of the data table.")
            
            # interpolation
            interpolated_data = {}
            for diameter in self.exp_data.index:

                row_data = self.exp_data.loc[diameter, time_points]
                ''' 
                # Compares the corresponding values ​​in time_point and row_data_index
                row_data_index = row_data.index
                
                for idx in range(len(time_points)):
                    time_point = time_points[idx]
                    row_data_time_point = row_data_index[idx]
                    
                    if time_point != row_data_time_point:
                        print(f"Nonzero differences for diameter {diameter} at index {idx}: {time_point} vs {row_data_time_point}")
                '''
                interp_func = interp1d(time_points, row_data, kind='linear')
                interpolated_data[diameter] = float(interp_func(t_exp))
            # fix a type bug for the first element in interpolated_data
            
            
            return pd.Series(interpolated_data, name=t_exp)
