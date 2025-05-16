# -*- coding: utf-8 -*-
"""
data-processing-related calculations during optimization
"""
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.interpolate import interp1d
from optframework.utils.func.func_read_exp import write_read_exp

def read_exp(self, exp_data_path, t_vec):  
    """
    Reads experimental data from a specified path and processes it for use in the optimization.

    Parameters
    ----------
    exp_data_path : str
        Path to the experimental data file.
    t_vec : array-like
        The time vector corresponding to the desired time points for the experimental data.

    Returns
    -------
    tuple of arrays
        - x_uni_exp: An array of unique particle sizes from the experimental data.
        - sumN_uni_exp: An array of the sum of number concentrations for the unique particle sizes.
    """
    # Instantiate the write_read_exp class to handle reading and writing PSD data,
    # and initialize the time format for the experimental data
    exp_data = write_read_exp(exp_data_path, read=True, sheet_name=self.sheet_name, exp_data=self.exp_data)
    
    # Extract the experimental data corresponding to the given time vector
    df = exp_data.get_exp_data(t_vec)
    
    # Get the particle sizes (x_uni_exp) and corresponding number concentrations (sumN_uni_exp)
    x_uni_exp = df.index.to_numpy()
    sumN_uni_exp = df.to_numpy()
    
    return x_uni_exp, sumN_uni_exp

def get_all_synth_data(self, exp_data_path):
    """
    Process synthetic data by reading the data and converting it to volume-based PSD.

    This method processes synthetic experimental data for one or multiple samples. For each sample, 
    it reads the experimental data, converts the number-based particle size distribution (PSD) into 
    a volume-based distribution, and then recalculates the final distribution (including q3, Q3, and 
    x_50) based on the specified flags. The results are stored in `data_exp` for use in subsequent 
    optimization steps.

    Parameters
    ----------
    exp_data_path : str
        Path to the experimental data file.

    Returns
    -------
    tuple of arrays
        - x_uni_exp: List of arrays containing unique particle sizes for each sample.
        - data_exp: List of arrays containing processed experimental PSD data for each sample.
    """
    # If only one sample exists, read and process the experimental data
    if self.sample_num == 1:
        x_uni_exp, sumN_uni_exp = self.read_exp(exp_data_path, self.t_vec[self.delta_t_start_step:]) 
        x_uni_exp = np.insert(x_uni_exp, 0, 0.0)
        sumN_uni_exp = np.insert(sumN_uni_exp, 0, 0.0, axis=0)
        
        # Convert number-based PSD to volume-based PSD
        vol_uni = np.tile((1/6)*np.pi*x_uni_exp**3, (self.num_t_steps-self.delta_t_start_step, 1)).T
        sumvol_uni_exp = sumN_uni_exp * vol_uni
        
        
        # Recalculate the distribution
        for flag, _ in self.delta_flag:
            data_exp = self.re_calc_distribution(x_uni_exp, sum_uni=sumvol_uni_exp, flag=flag)[0]

    # If multiple samples exist, process each one
    else:
        x_uni_exp = []
        data_exp = []
        for i in range (0, self.sample_num):
            # Read and process experimental data for each sample
            exp_data_path = self.traverse_path(i, exp_data_path)
            x_uni_exp_tem, sumN_uni_exp = self.read_exp(exp_data_path, self.t_vec[self.delta_t_start_step:])
            x_uni_exp_tem = np.insert(x_uni_exp_tem, 0, 0.0)
            sumN_uni_exp = np.insert(sumN_uni_exp, 0, 0.0, axis=0)
            
            # Convert number-based PSD to volume-based PSD
            vol_uni = np.tile((1/6)*np.pi*x_uni_exp_tem**3, (self.num_t_steps-self.delta_t_start_step, 1)).T
            sumvol_uni_exp = sumN_uni_exp * vol_uni
            
            # Recalculate the distribution
            for flag, _ in self.delta_flag:
                data_exp_tem = self.re_calc_distribution(x_uni_exp_tem, sum_uni=sumvol_uni_exp, flag=flag)[0] 
            x_uni_exp.append(x_uni_exp_tem)
            data_exp.append(data_exp_tem)
            
    return x_uni_exp, data_exp

## test only for 1d batch exp data
def get_all_exp_data(self, exp_data_path):
    if self.sample_num == 1:
        x_uni_exp, data_exp = self.read_exp(exp_data_path, self.t_vec[self.delta_t_start_step:]) 
        x_uni_exp = np.insert(x_uni_exp, 0, 0.0)
        zero_row = np.zeros((1, data_exp.shape[1]))
        data_exp = np.insert(data_exp, 0, zero_row, axis=0)
        
    else:
        x_uni_exp = []
        data_exp = []
        zero_row = np.zeros((1, data_exp.shape[1]))
        for i in range (0, self.sample_num):
            # Read and process experimental data for each sample
            exp_data_path = self.traverse_path(i, exp_data_path)
            x_uni_exp_tem, data_exp_tem = self.read_exp(exp_data_path, self.t_vec[self.delta_t_start_step:])
            x_uni_exp_tem = np.insert(x_uni_exp_tem, 0, 0.0)
            data_exp_tem = np.insert(data_exp_tem, 0, zero_row, axis=0)
            x_uni_exp.append(x_uni_exp_tem)
            data_exp.append(data_exp_tem)
        
    return x_uni_exp, data_exp

def function_noise(self, ori_data):
    """
    Adds noise to the original data based on the
    `noise_type` and `noise_strength` attributes. Supported noise types include 
    Gaussian ('Gaus'), Uniform ('Uni'), Poisson ('Po'), and Multiplicative ('Mul'). 
    The resulting noisy data is clipped to be non-negative.

    Parameters
    ----------
    ori_data : array-like
        The original data to which noise will be added.

    Returns
    -------
    array-like
        The noised data.

    Notes
    -----
    The noise types behave as follows:
    - Gaussian ('Gaus'): Adds noise with mean 0 and standard deviation `noise_strength`.
    - Uniform ('Uni'): Adds noise uniformly distributed over [-`noise_strength`/2, `noise_strength`/2).
    - Poisson ('Po'): Adds Poisson-distributed noise where `noise_strength` serves as lambda.
    - Multiplicative ('Mul'): Applies Gaussian multiplicative noise with mean 1 and standard deviation 
      `noise_strength`, multiplying the original data by the generated noise.

    The resulting noised data is clipped to ensure no negative values.
    """
    # Get the shape of the original data and initialize the noise array
    rows, cols = ori_data.shape
    noise = np.zeros((rows, cols))
    
    if self.noise_type == 'Gaus':
        # Add Gaussian noise with mean 0 and standard deviation `noise_strength`
        for i in range(cols):
            noise[:, i] = np.random.normal(0, self.noise_strength, rows)              
        noised_data = ori_data + noise
        
    elif self.noise_type == 'Uni':
        # Uniform noise over the interval [-`noise_strength`/2, `noise_strength`/2)
        for i in range(cols):
            noise[:, i] = np.random.uniform(low=-self.noise_strength/2, high=self.noise_strength/2, size=rows)
        noised_data = ori_data + noise
        
    elif self.noise_type == 'Po':
        # Poisson noise with `noise_strength` as lambda
        for i in range(cols):
            noise[:, i] = np.random.poisson(self.noise_strength, rows)
        noised_data = ori_data + noise
        
    elif self.noise_type == 'Mul':
        # Multiplicative Gaussian noise with mean 1 and standard deviation `noise_strength`
        for i in range(cols):
            noise[:, i] = np.random.normal(1, self.noise_strength, rows)
        noised_data = ori_data * noise
        
    # Ensure the data remains non-negative by clipping
    noised_data = np.clip(noised_data, 0, np.inf)
    return noised_data

def KDE_fit(self, x_uni_ori, data_ori, bandwidth='scott', kernel_func='epanechnikov'):
    """
    Fit a Kernel Density Estimation (KDE) model to the original data using the 
    specified kernel function and bandwidth. 

    Parameters
    ----------
    x_uni_ori : array-like
        The unique values of the data variable. Must be a one-dimensional array.
    data_ori : array-like
        The original data corresponding to `x_uni_ori`. Should be absolute values, not relative.
    bandwidth : float or {'scott', 'silverman'}, optional
        The bandwidth of the kernel. If a float is provided, it defines the bandwidth directly. 
        If a string ('scott' or 'silverman') is provided, the bandwidth is estimated using one 
        of these methods. Defaults to 'scott'.
    kernel_func : {'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'}, optional
        The kernel to use for the density estimation. Defaults to 'epanechnikov'.

    Returns
    -------
    sklearn.neighbors.kde.KernelDensity
        The fitted KDE model.
    
    Notes
    -----
    - `x_uni_ori` must be reshaped into a column vector for compatibility with the KernelDensity class.
    - Any values in `data_ori` that are zero or less are adjusted to a small positive value (1e-20) to 
      avoid numerical issues during KDE fitting.
    """  
    # Reshape the input data to be compatible with KernelDensity
    x_uni_ori_re = x_uni_ori.reshape(-1, 1)
    # Avoid divide-by-zero errors by adjusting very small or zero data points
    data_ori_adjested = np.where(data_ori <= 0, 1e-20, data_ori) 
    # Create and fit the KDE model with the specified kernel and bandwidth     
    kde = KernelDensity(kernel=kernel_func, bandwidth=bandwidth)
    kde.fit(x_uni_ori_re, sample_weight=data_ori_adjested)  
    return kde

def KDE_score(self, kde, x_uni_new):
    """
    Evaluate and normalize the KDE model on new data points based on the 
    cumulative distribution function (Q3).

    Parameters
    ----------
    kde : sklearn.neighbors.kde.KernelDensity
        The fitted KDE model from the :meth:`~.KDE_fit` method.
    x_uni_new : array-like
        New unique data points where the KDE model will be evaluated.

    Returns
    -------
    array-like
        The smoothed and normalized data based on the KDE model.

    Notes
    -----
    - The KDE model is evaluated on the new data points by calculating the log density, which is 
      then exponentiated to get the actual density values.
    - The smoothed data is normalized by dividing by the last value of the cumulative distribution (Q3).
    """
    # Reshape the new data points to match the input format expected by the KDE model
    x_uni_new_re = x_uni_new.reshape(-1, 1) 
    
    # Evaluate the KDE model to get the smoothed density values
    data_smoothing = np.exp(kde.score_samples(x_uni_new_re))
    
    # Flatten a column vector into a one-dimensional array
    data_smoothing = data_smoothing.ravel()
    
    # Normalize the smoothed data using the cumulative distribution function (Q3)
    Q3 = self.calc_Q3(x_uni_new, data_smoothing)
    data_smoothing = data_smoothing / Q3[-1]
    
    return data_smoothing

def traverse_path(self, label, path_ori):
    """
    Update the file path or list of file paths based on the given label.

    This method modifies the provided file path or a list of file paths by appending 
    or updating a numerical label (e.g., '_0', '_1') to distinguish different samples 
    of the same test.

    Parameters
    ----------
    label : int
        The label to update or append to the file path(s). The label corresponds to the 
        current sample or iteration number.
    path_ori : str or list of str
        The original file path or list of file paths to be updated.

    Returns
    -------
    str or list of str
        The updated file path(s) with the new label.
    """
    def update_path(path, label):
        # For label 0, append '_0' to the file name before the extension
        if label == 0:
            return path.replace(".xlsx", f"_{label}.xlsx")
        # For other labels, replace the previous label with the current label
        else:
            return path.replace(f"_{label-1}.xlsx", f"_{label}.xlsx")
        
    # If the input is a list of paths, update each path in the list
    if isinstance(path_ori, list):
        return [update_path(path, label) for path in path_ori]
    else:
        return update_path(path_ori, label)    
