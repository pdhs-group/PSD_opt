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
import statsmodels.formula.api as smf
from scipy.optimize   import minimize_scalar
import numdifftools as nd
from scipy.stats import chi2

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

def save_average_xQ_data(x_avg_array, Q_x_ref, sorted_time_labels):
    # Convert time labels from minutes to formatted string
    sorted_minutes = [extract_minutes(label) for label in sorted_time_labels]
    formatted_time_labels = [str(timedelta(minutes=minutes)) for minutes in sorted_minutes]

    # Create original x(Q) DataFrame
    x_Q_df = pd.DataFrame(x_avg_array.T, index=Q_x_ref, columns=formatted_time_labels)
    x_Q_df.index.name = 'Circular Equivalent Diameter'

    # Use mean x values as common interpolation points
    # x_m = x_avg_array.mean(axis=0)
    x_m = generate_nonuniform_grid(x_min=0.037, x_max=1, num_points=30, gamma=1.5)
    Q_x_int = np.zeros((x_avg_array.shape[0], len(x_m)))
    x_volume_mean =  np.zeros(x_avg_array.shape[0])

    # Interpolate Q(x), normalize to [0, 1]
    for i in range(x_avg_array.shape[0]):
        Q_x_int[i, 1:], x_volume_mean[i] = interpolate_Qx(
            x_vals=x_avg_array[i, :],
            Q_vals=Q_x_ref,
            x_target=x_m[1:],
            method=interpolation_method,     #'int1d', 'pchip', 'isotonic', 'lognormal'
            fraction=1
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
    return x_Q_df, Q_x_int_df, qx_int_df, x_volume_mean

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

def interpolate_Qx(x_vals, Q_vals, x_target, method="int1d", fraction=1):
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
    # elif method == "pchip":
    #     f_base = PchipInterpolator(x_sub, Q_sub, extrapolate=False)
    # elif method == "akima":
    #     f = Akima1DInterpolator(x_sub, Q_sub)
    # elif method == "spline_monotonic":
    #     # Smoothing spline with positive weights (approx monotonicity)
    #     f = UnivariateSpline(x_sub, Q_sub, s=1e-6)  # s 可调
    # elif method == "isotonic":
    #     ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='nan')
    #     Q_iso = ir.fit_transform(x_sub, Q_sub)
    #     f_base = interp1d(x_sub, Q_iso, bounds_error=False, fill_value=np.nan)
    elif method == "lognormal":
        f, (sigma, mu) = fit_lognormal_cdf(x_sub, Q_sub, method="curve_fit", weight_mode='uniform')
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    if method != "lognormal":
        f = wrap_with_linear_extrapolation(f_base, x_sub, Q_sub)
    Qx_vals = f(x_target)
    # Q_test = f(x_sub)
    # print(np.mean(abs(Q_sub-Q_test)))
    # Clip to [0, 1]
    # Qx_vals[Qx_vals < 0] = 0.0
    Qx_vals /= Qx_vals[-1]
    Qx_vals = np.clip(Qx_vals, 0.0, 1.0)
    if method != "lognormal":
        # dQ = np.diff(np.insert(Qx_vals, 0, 0.0))
        # x_volume_mean_val = np.sum(dQ*x_target**3)**(1/3)
        
        dx = np.diff(x_target)
        num = (Qx_vals[1:] - Qx_vals[:-1]) * (x_target[1:]**4 - x_target[:-1]**4) / (4.0*dx)
        x_volume_mean_val = np.sum(num)**(1/3)
    else:
        x_volume_mean_val = np.exp( mu + 1.5*sigma**2 )
    
    ## Warning: The calculation of x_volume_mean provided by interpolate_Qx is based on Q_vals as Q0.
    return Qx_vals, x_volume_mean_val

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
         
def calc_n_for_G(x_log_mean, G_flag):
    if G_flag == "Median_Integral":
        G_datas = [32.0404, 39.1135, 41.4924, 44.7977, 45.6443]
    elif G_flag == "Median_LocalStirrer":
        G_datas = [104.014, 258.081, 450.862, 623.357, 647.442]
    elif G_flag == "Mean_Integral":
        G_datas = [87.2642, 132.668, 143.68, 183.396, 185.225]
    elif G_flag == "Mean_LocalStirrer":
        G_datas = [297.136, 594.268, 890.721, 1167.74, 1284.46]
    else:
        raise ValueError(f"Unknown G_flag: {G_flag}")
        
    G_values = np.array(G_datas, dtype=float)
    times = np.array([0, 5, 10, 45], dtype=float)     
    x_mean_array = np.zeros((5, 4))
    for i, x_mean in enumerate(x_log_mean):
        x_mean_array[i, :] = x_mean[:4]
        
    records = []
    for i, G in enumerate(G_values):
        for j in range(1, len(times)):  # 5, 10, 45 min
            dt = times[j] - times[0]
            rate = - (x_mean_array[i, j] - x_mean_array[i, 0]) / dt
            # rate = - (x_mean_array[i, j] - x_mean_array[:, 0]).mean() / dt
            records.append({
                'group': i,
                'time': times[j],
                'G': G,
                'rate': rate
            })

    df = pd.DataFrame(records)
    df = df[df['rate'] > 0].copy()
    df['log_rate'] = np.log(df['rate'])
    df['log_G'] = np.log(df['G'])

    model = smf.ols("log_rate ~ log_G + C(time)", data=df).fit()
    print(model.summary())

    n_est = model.params['log_G']
    ci_lo, ci_hi = model.conf_int().loc['log_G']
    print(f"\nEstimated n = {n_est:.4f}  (95 % CI: {ci_lo:.4f} – {ci_hi:.4f})")

    # === Plotting ===
    plt.figure(figsize=(8, 6), dpi=150)
    time_markers = {5: 'o', 10: 's', 45: 'D'}
    time_colors = {5: 'tab:blue', 10: 'tab:orange', 45: 'tab:green'}

    # Plot scatter points
    for time_val in [5, 10, 45]:
        sub_df = df[df['time'] == time_val]
        plt.scatter(
            sub_df['log_G'], sub_df['log_rate'],
            label=f"{time_val} min",
            marker=time_markers[time_val],
            color=time_colors[time_val]
        )

    # Plot regression lines for each time
    x_fit = np.linspace(df['log_G'].min() - 0.2, df['log_G'].max() + 0.2, 200)

    # Time 5 min: baseline
    intercept_5 = model.params['Intercept']
    y_fit_5 = n_est * x_fit + intercept_5
    plt.plot(x_fit, y_fit_5, linestyle='--', color=time_colors[5], label="Fit: 5 min")

    # Time 10 min
    offset_10 = model.params.get('C(time)[T.10.0]', 0.0)
    intercept_10 = intercept_5 + offset_10
    y_fit_10 = n_est * x_fit + intercept_10
    plt.plot(x_fit, y_fit_10, linestyle='--', color=time_colors[10], label="Fit: 10 min")

    # Time 45 min
    offset_45 = model.params.get('C(time)[T.45.0]', 0.0)
    intercept_45 = intercept_5 + offset_45
    y_fit_45 = n_est * x_fit + intercept_45
    plt.plot(x_fit, y_fit_45, linestyle='--', color=time_colors[45], label="Fit: 45 min")

    plt.xlabel("log(G)")
    plt.ylabel("log(rate)")
    plt.title(f"log(rate) vs log(G) — {G_flag}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return df, x_mean_array
        
def calc_n_for_G_new(x_log_mean, G_flag):
    if G_flag == "Median_Integral":
        G_datas = [32.0404, 39.1135, 41.4924, 44.7977, 45.6443]
    elif G_flag == "Median_LocalStirrer":
        G_datas = [104.014, 258.081, 450.862, 623.357, 647.442]
    elif G_flag == "Mean_Integral":
        G_datas = [87.2642, 132.668, 143.68, 183.396, 185.225]
    elif G_flag == "Mean_LocalStirrer":
        G_datas = [297.136, 594.268, 890.721, 1167.74, 1284.46]
    else:
        raise ValueError(f"Unknown G_flag: {G_flag}")
        
    G_values = np.array(G_datas, dtype=float)
    times = np.array([5, 10, 45], dtype=float)    
    x_mean_array = np.zeros((5, 3))
    for i, x_mean in enumerate(x_log_mean):
        x_mean_array[i, :] = x_mean[1:4]
    
    ref_idx = 3
    t_ref = times
    D_ref = x_mean_array[ref_idx]
    G_ref = G_values[ref_idx]
    interp_ref = interp1d(t_ref, D_ref, kind='linear', fill_value='extrapolate')
    def sse(n):
        tot = 0.0
        # Construct the residual sum of squares function for all points (after stretching/compression) across all groups.
        for i in range(len(G_values)):
            if i == ref_idx: continue
            factor = (G_values[i]/G_ref)**n
            t_scaled = times * factor
            D_est = interp_ref(t_scaled)
            tot += np.sum((x_mean_array[i] - D_est)**2)
        return tot
    
    res = minimize_scalar(sse, bounds=(-5,5), method='bounded')
    
    n_star = res.x
    hessian_func = nd.Hessian(sse)
    second_derivative = hessian_func([n_star])[0, 0]
    alpha = 0.95
    delta_S = chi2.ppf(alpha, df=1)
    half_width = np.sqrt(2 * delta_S / second_derivative)
    ci_lower = n_star - half_width
    ci_upper = n_star + half_width
    
    print(f"\nEstimated n = {n_star:.4f}  (95 % CI: {ci_lower:.4f} – {ci_upper:.4f})")
    return res.x, res.fun

def Q0_to_Q3(d, Q0, use_bin_average_volume=False):
    Q0_ext = np.concatenate([[0.0], Q0])
    dQ0    = np.diff(Q0_ext)               # length m
    
    if use_bin_average_volume:
        d3_prev = np.concatenate([[0.0], d**3])[:-1]
        v_rep   = (d3_prev + d**3) / 2 * (np.pi/6)
    else:
        v_rep   = (np.pi/6) * d**3
    
    vol_bin   = dQ0 * v_rep
    total_vol = vol_bin.sum()
    if total_vol <= 0:
        raise ValueError("total volume is zero!")
    dQ3 = vol_bin / total_vol       
    Q3 = np.cumsum(dQ3)
    
    return Q3

if __name__ == '__main__':
    len_data = 201 
    interpolation_method = "int1d"     #'int1d', 'pchip', 'isotonic', 'lognormal'
    # fit_lognormal_method = "curve_fit" #'curve_fit', 'zscore', 'global_opt'
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
    for raw_file in batch_files:
        filename_base, ext = os.path.splitext(os.path.basename(raw_file))
        file_path = os.path.join(base_path, raw_file)
        data, measurement_count = load_excel_data(file_path)
        x_avg_array, Q_x_ref, sorted_time_labels = process_data_xQ(data, measurement_count)
        x_int_list.append(x_avg_array[0,:])
        
    x_int_avg = np.mean(x_int_list, axis=0)    
    
    Q_x_int_df_list = []
    Vmean_list = []
    Vmean2_list = []
    x_log_mean = []
    for raw_file in batch_files:
        filename_base, ext = os.path.splitext(os.path.basename(raw_file))
        file_path = os.path.join(base_path, raw_file)
        data, measurement_count = load_excel_data(file_path)
        x_avg_array, Q_x_ref, sorted_time_labels = process_data_xQ(data, measurement_count)
        x_avg_array[0, :] = x_int_avg
        x_Q_df, Q_x_int_df, qx_int_df, x_volume_mean = save_average_xQ_data(x_avg_array, Q_x_ref, sorted_time_labels)
        Q_x_int_df_list.append(Q_x_int_df)
        plot_xQ_profiles(x_Q_df, Q_x_int_df, qx_int_df)
        
        # right side integration with original data
        # dQ = np.diff(np.insert(Q_x_ref, 0, 0.0))
        # Vmean = (np.pi/6.0) * np.sum( dQ * x_avg_array**3, axis=1 )
        # Vmean_list.append(Vmean)
        # hight precise with original data
        # dx  = np.diff(x_avg_array)
        # num = (Q_x_ref[1:] - Q_x_ref[:-1]) * (x_avg_array[:, 1:]**4 - x_avg_array[:, :-1]**4) / (4.0*dx)
        # Vmean2 = (np.pi/6.0) * np.sum(num, axis=1)
        # Vmean2_list.append(Vmean2)
        
        
        # log-normal form
        x_log_mean.append(x_volume_mean)
        
        
    # plot_Qx_time_G_profiles(Q_x_int_df_list)
    
    # x_int_avg = np.mean(x_int_list, axis=0) * 1e-6     # convert the unit from um to m
    # Q3_ref = Q0_to_Q3(x_int_avg, Q_x_ref, False)
    # dict_Qx={'Q_PSD':Q3_ref,'x_PSD':x_int_avg, 'r0_001':x_int_avg[0], 'r0_005':x_int_avg[1], 'r0_01':x_int_avg[2]}
    
    # Q_columns = [df["0:00:00"] for df in Q_x_int_df_list]
    # Q_concat = pd.concat(Q_columns, axis=1)
    # Q_x_int_mean = np.array(Q_concat.mean(axis=1))
    # x_int = np.array(Q_concat.index) * 1e-6
    # Q3_ref = Q0_to_Q3(x_int, Q_x_int_mean, False)
    # dict_Qx={'Q_PSD':Q3_ref,'x_PSD':x_int, 'r0_001':x_int[0], 'r0_005':x_int[1], 'r0_01':x_int[2]}
    
    # dist_path = os.path.join(base_path, "Batch_int_PSD.npy")
    # np.save(dist_path,dict_Qx)
    
    G_flag = "Mean_LocalStirrer"
    df, x_mean_array = calc_n_for_G_new(x_log_mean, G_flag)
    
    
    

