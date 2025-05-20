# -*- coding: utf-8 -*-
"""
Convert experimental data into data that can be directly input into opt
"""
import os
import re
import math
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, PchipInterpolator, Akima1DInterpolator, UnivariateSpline
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from scipy.stats import norm
from scipy.optimize import curve_fit, differential_evolution
from scipy.special import erf

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
    
    excel_data.close()
    
    return merged_data, measurement_count

def extract_minutes(label):
    match = re.search(r'(\d+) min', label)
    return int(match.group(1)) - 15  if match else 0

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

# def process_data(merged_data, measurement_count):
#     # Sort the time labels to ensure chronological order
#     sorted_time_labels = sorted(merged_data.keys(), key=extract_minutes)

#     # Create 3D arrays for each type of measurement
#     Q_x_array = np.zeros((measurement_count, len(sorted_time_labels), len_data))
#     q_lnx_array = np.zeros((measurement_count, len(sorted_time_labels), len_data))
#     q_x_array = np.zeros((measurement_count, len(sorted_time_labels), len_data))
#     x_array = np.zeros((measurement_count, len(sorted_time_labels), len_data))
#     xm_array = np.zeros((measurement_count, len(sorted_time_labels), len_data))
    
#     for time_label in sorted_time_labels:
#         measurements = merged_data[time_label]
#         Q_x = measurements['Q(x)']
#         q_lnx = measurements['q(lnx)']
#         x = measurements['x']
#         xm = measurements['xm']

#         for i in range(measurement_count):
#             Q_x_array[i, sorted_time_labels.index(time_label), :] = Q_x[i]
#             q_lnx_array[i, sorted_time_labels.index(time_label), :] = q_lnx[i]
#             x_array[i, sorted_time_labels.index(time_label), :] = x[i]
#             xm_array[i, sorted_time_labels.index(time_label), :] = xm[i]
#     q_x_array[:,:,1:] = q_lnx_array[:,:,1:] / xm_array[:,:,1:]
#     # Find the global min and max values for x and xm
#     x_min, x_max = np.min(x_array), np.max(x_array)
#     xm_min, xm_max = np.min(xm_array[np.nonzero(xm_array)]), np.max(xm_array)
    
#     # Generate new coordinates
#     new_xm_coords = np.zeros(200)
#     new_x_coords = generate_nonuniform_coords(x_min, x_max, 200)
#     new_xm_coords[1:] = generate_nonuniform_coords(xm_min, xm_max, 199)
#     Q_x_int_array = interpolate_data(Q_x_array, x_array, new_x_coords)
#     q_x_int_array = interpolate_data(q_x_array, xm_array, new_xm_coords)
#     ## Theoretically, q_x also needs to be normalized, 
#     ## but it needs to be integrated, which will cause a larger error!!!
#     for i in range(Q_x_int_array.shape[0]):
#         for j in range(Q_x_int_array.shape[1]):
#             scale = Q_x_int_array[i,j,199]
#             Q_x_int_array[i,j,:] /= scale
    
#     return Q_x_int_array, q_x_int_array, new_x_coords, new_xm_coords, sorted_time_labels

def process_data_xQ(merged_data, measurement_count):
    # Sort the time labels to ensure chronological order
    sorted_time_labels = sorted(merged_data.keys(), key=extract_minutes)

    # Create 3D arrays for each type of measurement
    Q_x_array = np.zeros((measurement_count, len(sorted_time_labels), len_data))
    x_array = np.zeros((measurement_count, len(sorted_time_labels), len_data))
    
    for time_label in sorted_time_labels:
        measurements = merged_data[time_label]
        Q_x = measurements['Q(x)']
        x = measurements['x']

        for i in range(measurement_count):
            Q_x_array[i, sorted_time_labels.index(time_label), :] = Q_x[i]
            x_array[i, sorted_time_labels.index(time_label), :] = x[i]
            
    x_avg_array = np.mean(x_array, axis=0)
    Q_x_ref = Q_x_array[0, 0, :]
    
    return x_avg_array, Q_x_ref, sorted_time_labels

# def save_interpolated_data(Q_x_int_array, q_x_int_array, x_arrays, xm_arrays, sorted_time_labels, measurement_count):
#     # Extract minutes from sorted_time_labels and convert to '%H:%M:%S' format
#     sorted_minutes = [extract_minutes(label) for label in sorted_time_labels]
#     formatted_time_labels = [str(timedelta(minutes=minutes)) for minutes in sorted_minutes]
    
#     # Create a list to store dataframes
#     Q_x_dfs = []
#     q_x_dfs = []

#     for i in range(measurement_count):
#         # Create dataframe for Q_x_int_array
#         Q_x_df = pd.DataFrame(Q_x_int_array[i].T, index=x_arrays, columns=formatted_time_labels)
#         Q_x_df.index.name = 'Circular Equivalent Diameter'
#         Q_x_dfs.append(Q_x_df)

#         # Create dataframe for q_x_int_array
#         q_x_df = pd.DataFrame(q_x_int_array[i].T, index=xm_arrays, columns=formatted_time_labels)
#         q_x_df.index.name = 'Circular Equivalent Diameter'
#         q_x_dfs.append(q_x_df)
#         save_path = os.path.join(r"C:\Users\px2030\Code\Ergebnisse\BatchDaten\post", f"Batch_600_Q3_Graphite_{i}.xlsx")
#         with pd.ExcelWriter(save_path) as writer:
#             Q_x_df.to_excel(writer, sheet_name='Q_x')
#             q_x_df.to_excel(writer, sheet_name='q_x')
#     return Q_x_dfs, q_x_dfs

def save_average_xQ_data_old(x_avg_array, Q_x_ref, sorted_time_labels):
    # Extract minutes from sorted_time_labels and convert to '%H:%M:%S' format
    sorted_minutes = [extract_minutes(label) for label in sorted_time_labels]
    formatted_time_labels = [str(timedelta(minutes=minutes)) for minutes in sorted_minutes]
    
    # Create dataframe for Q_x_int_array
    x_Q_df = pd.DataFrame(x_avg_array.T, index=Q_x_ref, columns=formatted_time_labels)
    x_Q_df.index.name = 'Circular Equivalent Diameter'
    
    x_m = np.zeros_like(x_avg_array)
    qx = np.zeros_like(x_avg_array)
    x_m[:, 1:] = (x_avg_array[:, :-1] + x_avg_array[:, 1:]) / 2
    for i in range(x_m.shape[0]):
        qx[i, 1:] = (Q_x_ref[1:] - Q_x_ref[:-1]) / (x_avg_array[i, 1:] - x_avg_array[i, :-1])
    xmm = x_m.mean(axis=0)    
    q_x_df = pd.DataFrame(qx.T, index=xmm, columns=formatted_time_labels)    
    q_x_df.index.name = 'Circular Equivalent Diameter'
    
    qx_int = np.zeros((qx.shape[0], len(xmm)))
    x_mmm = x_avg_array.mean(axis=0)
    for i in range(x_m.shape[0]):
        f = interp1d(x_m[i, 1:], qx[i, 1:], bounds_error=False, fill_value="extrapolate")
        qx_int[i, 1:] = f(xmm[1:])
    qx_int[np.where(qx_int<0)] = 0.0
    for i in range(x_m.shape[0]):
        qx_sum = sum(qx_int[i, 1:] * (x_mmm[1:] - x_mmm[:-1]))
        qx_int[i, 1:] = qx_int[i, 1:] / qx_sum
    qx_int_df = pd.DataFrame(qx_int.T, index=xmm, columns=formatted_time_labels)    
    qx_int_df.index.name = 'Circular Equivalent Diameter'
        
    save_path = os.path.join(r"C:\Users\px2030\Code\Ergebnisse\BatchDaten\post", "Batch_600_Q0_post.xlsx")
    with pd.ExcelWriter(save_path) as writer:
        x_Q_df.to_excel(writer, sheet_name='Q_x')
        q_x_df.to_excel(writer, sheet_name='q_x')
        qx_int_df.to_excel(writer, sheet_name='q_x_int')
    return x_Q_df

def save_average_xQ_data(x_avg_array, Q_x_ref, sorted_time_labels):
    # Convert time labels from minutes to formatted string
    sorted_minutes = [extract_minutes(label) for label in sorted_time_labels]
    formatted_time_labels = [str(timedelta(minutes=minutes)) for minutes in sorted_minutes]

    # Create original x(Q) DataFrame
    x_Q_df = pd.DataFrame(x_avg_array.T, index=Q_x_ref, columns=formatted_time_labels)
    x_Q_df.index.name = 'Circular Equivalent Diameter'

    # Use mean x values as common interpolation points
    # x_m = x_avg_array.mean(axis=0)
    x_m = generate_nonuniform_grid(x_min=0.037, x_max=1, num_points=101, gamma=1.2)
    Q_x_int = np.zeros((x_avg_array.shape[0], len(x_m)))

    # Interpolate Q(x), normalize to [0, 1]
    for i in range(x_avg_array.shape[0]):
        Q_x_int[i, 1:] = interpolate_Qx(
            x_vals=x_avg_array[i, :],
            Q_vals=Q_x_ref,
            x_target=x_m[1:],
            method=interpolation_method,     #'int1d', 'pchip', 'isotonic', 'lognormal'
            fraction=1.0
        )

    Q_x_int_df = pd.DataFrame(Q_x_int[:, 1:].T, index=x_m[1:], columns=formatted_time_labels)
    Q_x_int_df.index.name = 'Circular Equivalent Diameter'

    # Compute q(x) by differentiating Q(x)
    qx_int = np.zeros_like(Q_x_int)
    for i in range(Q_x_int.shape[0]):
        dq = np.diff(Q_x_int[i, :])
        dx = np.diff(x_m)
        q = np.concatenate([[0], dq / dx])
        q[q < 0] = 0.0
        area = np.sum(q[1:] * dx)
        q[1:] = q[1:] / area if area != 0 else q[1:]
        qx_int[i, :] = q

    # Construct new x_mm for q(x): [0] + midpoints of intervals
    # x_mm = np.zeros_like(x_m)
    # x_mm[1:] = (x_m[:-1] + x_m[1:]) / 2
    x_mm = x_m

    qx_int_df = pd.DataFrame(qx_int[:, 1:].T, index=x_mm[1:], columns=formatted_time_labels)
    qx_int_df.index.name = 'Circular Equivalent Diameter'

    # Save Excel
    post_file = f"{filename_base}_post{ext}"
    save_path = os.path.join(base_path, "post", post_file)
    with pd.ExcelWriter(save_path) as writer:
        x_Q_df.to_excel(writer, sheet_name='Q_x')
        Q_x_int_df.to_excel(writer, sheet_name='Q_x_int')
        qx_int_df.to_excel(writer, sheet_name='q_x_int')

    # Optional: return for external usage
    return x_Q_df, Q_x_int_df, qx_int_df

def generate_nonuniform_grid(x_min, x_max, num_points, gamma=2.0, reverse=False):
    """
    Generate a non-uniform grid on [x_min, x_max], with controllable density.
    
    Parameters:
        x_min (float): Minimum of the grid
        x_max (float): Maximum of the grid
        num_points (int): Number of grid points
        gamma (float): Controls density: >1 = dense at start, <1 = dense at end
        reverse (bool): If True, reverse the grid (dense at max side)
    
    Returns:
        numpy.ndarray: Non-uniform grid array
    """
    # u = np.linspace(0, 1, num_points)
    # if reverse:
    #     u = 1 - u
    # s = u ** gamma
    # x = x_min + (x_max - x_min) * s
    V = np.zeros(num_points)
    x = np.zeros(num_points)
    V_e = np.zeros(num_points+1)
    V_e[0] = -4 * math.pi * x_min**3 / 3
    for i in range(num_points):
        V_e[i+1] = gamma**(i) * 4 * math.pi * x_min**3 / 3
        V[i] = (V_e[i] + V_e[i+1]) / 2
    x[1:] = (V[1:] * 3 / (4 * math.pi))**(1/3)
    
    return x

def interpolate_Qx(x_vals, Q_vals, x_target, method="pchip", fraction=0.9):
    """
    Interpolate Q(x) with various methods, and clip to [0, 1].

    Parameters:
        x_vals (1D array): Original x coordinates for Q(x)
        Q_vals (1D array): Corresponding Q(x) values
        x_target (1D array): Interpolation target points
        method (str): One of 'int1d', 'pchip', 'akima', 'spline_monotonic'
        fraction (float): Portion of data (0 < fraction <= 1.0) to use for interpolation

    Returns:
        qx_interp (1D array): Interpolated and clipped Q(x) values on x_target
    """
    assert 0 < fraction <= 1.0, "fraction must be between 0 and 1"
    N = int(len(x_vals) * fraction)
    x_sub = x_vals[:N]
    Q_sub = Q_vals[:N]

    if method == "int1d":
        f_base  = interp1d(x_sub, Q_sub, bounds_error=False, fill_value=np.nan)
    elif method == "pchip":
        f_base = PchipInterpolator(x_sub, Q_sub, extrapolate=False)
    # elif method == "akima":
    #     f = Akima1DInterpolator(x_sub, Q_sub)
    # elif method == "spline_monotonic":
    #     # Smoothing spline with positive weights (approx monotonicity)
    #     f = UnivariateSpline(x_sub, Q_sub, s=1e-6)  # s 可调
    elif method == "isotonic":
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='nan')
        Q_iso = ir.fit_transform(x_sub, Q_sub)
        f_base = interp1d(x_sub, Q_iso, bounds_error=False, fill_value=np.nan)
    elif method == "lognormal":
        f, _ = fit_lognormal_cdf(x_sub, Q_sub, method=fit_lognormal_method, weight_mode='uniform')
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    if method != "lognormal":
        f = wrap_with_linear_extrapolation(f_base, x_sub, Q_sub)
    Qx_vals = f(x_target)
    # Q_test = f(x_sub)
    # print(np.mean(abs(Q_sub-Q_test)))
    # Clip to [0, 1]
    # Qx_vals[Qx_vals < 0] = 0.0
    # Qx_vals /= Qx_vals[-1]
    Qx_vals = np.clip(Qx_vals, 0.0, 1.0)

    return Qx_vals

def fit_lognormal_cdf(x, Q, method='zscore', weight_mode='uniform', clip_eps=1e-6):
    """
    Fit a log-normal CDF to (x, Q) data using one of three methods:
    - 'curve_fit': Nonlinear least squares
    - 'zscore': Inverse normal CDF + linear regression
    - 'global_opt': Global optimization (DE) with optional tail weighting

    Parameters:
        x (array-like): Positive values (e.g., particle size)
        Q (array-like): Cumulative values (0–1 range)
        method (str): One of ['curve_fit', 'zscore', 'global_opt']
        weight_mode (str): 'uniform', 'tail', 'head', 'center'
        clip_eps (float): Min/max clipping for numerical stability

    Returns:
        f(x): Fitted CDF function (callable)
        (σ, μ): Log-normal parameters
    """
    x = np.asarray(x)
    Q = np.asarray(Q)

    assert np.all(x > 0), "x must be strictly positive for log-normal fit"
    assert 0 < clip_eps < 0.1

    # Define CDF model
    def lognorm_cdf(x, sigma, mu):
        ## log-normal distribution
        cdf = 0.5 * (1 + erf((np.log(x) - mu) / (sigma * np.sqrt(2))))
        return cdf

    if method == 'curve_fit':
        # Initial guess: sigma=1, mu=log(median x)
        p0 = [1.0, np.log(np.median(x))]
        bounds = ([1e-6, -np.inf], [np.inf, np.inf])
        popt, _ = curve_fit(lognorm_cdf, x, Q, p0=p0, bounds=bounds)
        sigma, mu = popt
        f = lambda xt: lognorm_cdf(xt, sigma, mu)

    elif method == 'zscore':
        # Inverse transform: norm.ppf(Q) ≈ (log x - mu) / sigma
        Qc = np.clip(Q, clip_eps, 1 - clip_eps)
        z = norm.ppf(Qc)
        y = np.log(x)
        sigma, mu = np.polyfit(z, y, 1)
        f = lambda xt: norm.cdf((np.log(xt) - mu) / sigma)

    elif method == 'global_opt':
        # Weight generation
        def get_weights(n, mode):
            if mode == 'uniform':
                return np.ones(n)
            elif mode == 'tail':
                return np.linspace(1, 5, n)
            elif mode == 'head':
                return np.linspace(5, 1, n)
            elif mode == 'center':
                mid = n // 2
                return -np.abs(np.arange(n) - mid) + (n // 2) + 1
            else:
                raise ValueError(f"Unknown weight mode: {mode}")
        weights = get_weights(len(x), weight_mode)
        def objective(params):
            sigma, mu = params
            model = lognorm_cdf(x, sigma, mu)
            return np.sum(weights * (model - Q) ** 2)

        bounds = [(1e-6, 3.0), (np.log(x.min()), np.log(x.max()))]
        res = differential_evolution(objective, bounds)
        sigma, mu = res.x
        f = lambda xt: lognorm_cdf(xt, sigma, mu)

    else:
        raise ValueError(f"Unknown method: {method}")

    return f, (sigma, mu)

def wrap_with_linear_extrapolation(f_raw, x_sub, y_sub):
    """
    Wrap a given interpolator with linear extrapolation on both ends.
    
    Parameters:
        f_raw (callable): Base interpolator function f(x)
        x_sub, y_sub (array-like): Original data points used for interpolation

    Returns:
        f(x): Interpolated + linearly extrapolated function
    """
    x_sub = np.asarray(x_sub)
    y_sub = np.asarray(y_sub)

    # Compute boundary slopes
    slope_left  = (y_sub[1] - y_sub[0]) / (x_sub[1] - x_sub[0])
    slope_right = (y_sub[-1] - y_sub[-2]) / (x_sub[-1] - x_sub[-2])
    x0, y0 = x_sub[0], y_sub[0]
    x1, y1 = x_sub[-1], y_sub[-1]

    def f(x):
        x = np.asarray(x)
        y = f_raw(x)

        mask_left  = x < x0
        mask_right = x > x1

        # Apply linear extrapolation on both ends
        y[mask_left]  = y0 + slope_left  * (x[mask_left]  - x0)
        y[mask_right] = y1 + slope_right * (x[mask_right] - x1)

        return y

    return f


def plot_xQ_profiles(x_Q_df, Q_x_int_df, qx_int_df):
    plt.figure(figsize=(7, 5))
    for label in x_Q_df.columns:
        plt.plot(x_Q_df[label], x_Q_df.index, label=label, linewidth=1)
    plt.xscale('log')
    plt.xlabel("x")
    plt.ylabel("Q")
    plt.title("x(Q) profiles over time")
    plt.legend(fontsize="small", loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    for label in Q_x_int_df.columns:
        x_loc = Q_x_int_df.index
        y_loc = Q_x_int_df[label]
        plt.plot(x_loc, y_loc, label=label, linewidth=1)
    plt.xscale('log')
    plt.xlabel("x")
    plt.ylabel("Q(x)")
    plt.title("Q(x) profiles over time")
    plt.legend(fontsize="small", loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    for label in qx_int_df.columns:
        x_loc = qx_int_df.index
        y_loc = qx_int_df[label]
        plt.plot(x_loc, y_loc, label=label, linewidth=1)
    plt.xscale('log')
    plt.xlabel("x")
    plt.ylabel("q(x)")
    plt.title("q(x) profiles over time")
    plt.legend(fontsize="small", loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_Qx_time_G_profiles(Q_x_int_df_list):
    for label in Q_x_int_df_list[1].columns:
        plt.figure(figsize=(7, 5))
        for i, Q_x_int_df in enumerate(Q_x_int_df_list):
            x_loc = Q_x_int_df.index
            y_loc = Q_x_int_df[label]
            plt.plot(x_loc, y_loc, label=batch_files[i], linewidth=1)
        plt.xscale('log')
        plt.xlabel("x")
        plt.ylabel("Q(x)")
        plt.title(f"Q(x) profiles over data at {label}")
        plt.legend(fontsize="small", loc="upper right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
            
if __name__ == '__main__':
    len_data = 201 
    interpolation_method = "int1d"     #'int1d', 'pchip', 'isotonic', 'lognormal'
    fit_lognormal_method = "curve_fit" #'curve_fit', 'zscore', 'global_opt'
    # Usage
    base_path = r"C:\Users\px2030\Code\Ergebnisse\BatchDaten"
    # raw_file = "Batch_1800_Q0.xlsx"
    batch_files = [
        "Batch_600_Q0.xlsx",
        "Batch_900_Q0.xlsx",
        "Batch_1200_Q0.xlsx",
        "Batch_1500_Q0.xlsx",
        "Batch_1800_Q0.xlsx",
    ]
    x_int_list = []
    Q_x_int_df_list = []
    for raw_file in batch_files:
        filename_base, ext = os.path.splitext(os.path.basename(raw_file))
        file_path = os.path.join(base_path, raw_file)
        data, measurement_count = load_excel_data(file_path)
        # Q_x_arrays, q_x_array, x_arrays, xm_arrays, sorted_time_labels = process_data(data, measurement_count)
        # Q_x_dfs, q_x_dfs = save_interpolated_data(Q_x_arrays, q_x_array, x_arrays, xm_arrays, sorted_time_labels,measurement_count)
        # Now `data` contains all the extracted information
        
        x_avg_array, Q_x_ref, sorted_time_labels = process_data_xQ(data, measurement_count)
        x_int_list.append(x_avg_array[0,:])
        x_Q_df, Q_x_int_df, qx_int_df = save_average_xQ_data(x_avg_array, Q_x_ref, sorted_time_labels)
        Q_x_int_df_list.append(Q_x_int_df)
        plot_xQ_profiles(x_Q_df, Q_x_int_df, qx_int_df)
        
    plot_Qx_time_G_profiles(Q_x_int_df_list)
    
    x_int_avg = np.mean(x_int_list, axis=0) * 1e-6     # convert the unit from um to m
    dist_path = os.path.join(base_path, "Batch_int_PSD.npy")
    dict_Qx={'Q_PSD':Q_x_ref,'x_PSD':x_int_avg, 'r0_001':x_int_avg[0], 'r0_005':x_int_avg[1], 'r0_01':x_int_avg[2]}
    np.save(dist_path,dict_Qx)

