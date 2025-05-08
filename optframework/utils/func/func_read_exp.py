# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:59:35 2023

Read or interpolate experimental data based on the time step of the simulated data

@author: px2030
"""
import os
import pandas as pd
from datetime import datetime, timedelta
from scipy.interpolate import interp1d

class write_read_exp():
    
    def __init__(self, path_exp_data = None, read = False, sheet_name=None, exp_data=False):
        
        if read:
            # read the original data
            self.file_name = os.path.basename(path_exp_data)
            if sheet_name is None:
                self.exp_data = pd.read_excel(path_exp_data)
            else:
                self.exp_data = pd.read_excel(path_exp_data, sheet_name=sheet_name)
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
                
        total_second = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1e6 
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
        
    def get_exp_data(self, t_vec):
        interpolated_results = {}
        
        for t_exp in t_vec:
            # use interpolation to find experiment data
            time_points = self.exp_data.columns[:]  
            
            # make sure that t_exp not out of the range
            # if t_exp < min(time_points) or t_exp > max(time_points):
            if t_exp < 0 or t_exp > max(time_points):
                raise ValueError(f"The experimental time is out of the range of the data file {self.file_name}.")
            
            # interpolation for time
            interpolated_data = {}
            for diameter in self.exp_data.index:

                row_data = self.exp_data.loc[diameter, time_points]
                interp_func = interp1d(time_points, row_data, kind='linear')
                interpolated_data[diameter] = float(interp_func(t_exp))
                
            interpolated_results[t_exp] = pd.Series(interpolated_data, name=t_exp)
        # convert the result in dataframe    
        result_df = pd.DataFrame(interpolated_results)
        return result_df
