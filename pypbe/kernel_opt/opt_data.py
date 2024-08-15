# -*- coding: utf-8 -*-
"""
data-processing-related calculations during optimization
"""
import numpy as np
from sklearn.neighbors import KernelDensity
from ..utils.func.func_read_exp import write_read_exp

## Read the experimental data and re-interpolate the particle distribution 
## of the experimental data according to the simulation results.
def read_exp(self, exp_data_path, t_vec):  
    """
    Reads experimental data from a specified path and processes it.

    Parameters
    ----------
    exp_data_path : `str`
        Path to the experimental data.

    Returns
    -------
    `tuple of array`
        - `x_uni_exp`: An array of unique particle sizes.
        - `sumN_uni_exp`: An array of sum of number concentrations for the unique particle sizes.
    """
    exp_data = write_read_exp(exp_data_path, read=True, sheet_name=self.sheet_name)
    df = exp_data.get_exp_data(t_vec)
    x_uni_exp = df.index.to_numpy()
    sumN_uni_exp = df.to_numpy()
    return x_uni_exp, sumN_uni_exp

def get_all_synth_data(self, exp_data_path):
    if self.sample_num == 1:
        x_uni_exp, sumN_uni_exp = self.read_exp(exp_data_path, self.t_vec[self.delta_t_start_step:]) 
        x_uni_exp = np.insert(x_uni_exp, 0, 0.0)
        sumN_uni_exp = np.insert(sumN_uni_exp, 0, 0.0, axis=0)
        vol_uni = np.tile((1/6)*np.pi*x_uni_exp**3, (self.num_t_steps-self.delta_t_start_step, 1)).T
        sumvol_uni_exp = sumN_uni_exp * vol_uni
        sumvol_uni_exp = np.insert(sumvol_uni_exp, 0, 0.0, axis=0)
        x_uni_exp = np.insert(x_uni_exp, 0, 0.0)
        for flag, _ in self.delta_flag:
            data_exp = self.re_calc_distribution(x_uni_exp, sum_uni=sumvol_uni_exp, flag=flag)[0]

    else:
        x_uni_exp = []
        data_exp = []
        for i in range (0, self.sample_num):
            exp_data_path = self.traverse_path(i, exp_data_path)
            x_uni_exp_tem, sumN_uni_exp = self.read_exp(exp_data_path, self.t_vec[self.delta_t_start_step:])
            x_uni_exp_tem = np.insert(x_uni_exp_tem, 0, 0.0)
            sumN_uni_exp = np.insert(sumN_uni_exp, 0, 0.0, axis=0)
            vol_uni = np.tile((1/6)*np.pi*x_uni_exp_tem**3, (self.num_t_steps-self.delta_t_start_step, 1)).T
            sumvol_uni_exp = sumN_uni_exp * vol_uni
            for flag, _ in self.delta_flag:
                data_exp_tem = self.re_calc_distribution(x_uni_exp_tem, sum_uni=sumvol_uni_exp, flag=flag)[0] 
            x_uni_exp.append(x_uni_exp_tem)
            data_exp.append(data_exp_tem)
    return x_uni_exp, data_exp

def get_all_exp_data(self, exp_data_path):
    if self.sample_num == 1:
        x_uni_exp, q3_exp = self.read_exp(exp_data_path, self.t_vec[self.delta_t_start_step:]) 
        data_exp = q3_exp
    else:
        x_uni_exp = []
        data_exp = []
        for i in range (0, self.sample_num):
            exp_data_path = self.traverse_path(i, exp_data_path)
            x_uni_exp_tem, q3_exp = self.read_exp(exp_data_path, self.t_vec[self.delta_t_start_step:])
            x_uni_exp.append(x_uni_exp_tem)
            data_exp.append(q3_exp)
    return x_uni_exp, data_exp

def function_noise(self, ori_data):
    """
    Adds noise to the original data based on the specified noise type.
    
    The method supports four types of noise: Gaussian ('Gaus'), Uniform ('Uni'), 
    Poisson ('Po'), and Multiplicative ('Mul'). The type of noise and its strength 
    are determined by the `noise_type` and `noise_strength` attributes, respectively.
    
    Parameters
    ----------
    ori_data : `array`
        The original data to which noise will be added.
    
    Returns
    -------
    `array`
        The noised data.
    
    Notes
    -----
    - Gaussian noise is added with mean 0 and standard deviation `noise_strength`.
    - Uniform noise is distributed over [-`noise_strength`/2, `noise_strength`/2).
    - Poisson noise uses `noise_strength` as lambda (expected value of interval).
    - Multiplicative noise is Gaussian with mean 1 and standard deviation `noise_strength`,
      and it is multiplied by the original data instead of being added.
    """
    rows, cols = ori_data.shape
    noise = np.zeros((rows, cols))
    if self.noise_type == 'Gaus':
        # The first parameter 0 represents the mean value of the noise, 
        # the second parameter is the standard deviation of the noise,
        for i in range(cols):
            noise[:, i] = np.random.normal(0, self.noise_strength, rows)              
        noised_data = ori_data + noise
        
    elif self.noise_type == 'Uni':
        # Noises are uniformly distributed over the half-open interval [low, high)
        for i in range(cols):
            noise[:, i] = np.random.uniform(low=-self.noise_strength/2, high=self.noise_strength/2, size=rows)
        noised_data = ori_data + noise
        
    elif self.noise_type == 'Po':
        for i in range(cols):
            noise[:, i] = np.random.poisson(self.noise_strength, rows)
        noised_data = ori_data + noise
        
    elif self.noise_type == 'Mul':
        for i in range(cols):
            noise[:, i] = np.random.normal(1, self.noise_strength, rows)
        noised_data = ori_data * noise
    # Cliping the data out of range  
    noised_data = np.clip(noised_data, 0, np.inf)
    return noised_data

## Kernel density estimation
## data_ori must be a quantity rather than a relative value!
def KDE_fit(self, x_uni_ori, data_ori, bandwidth='scott', kernel_func='epanechnikov'):
    """
    Fit a Kernel Density Estimation (KDE) to the original data.
    
    This method applies KDE to the original data using specified bandwidth and kernel function. 
    
    Parameters
    ----------
    x_uni_ori : `array`
        The unique values of the data variable, must be a one-dimensional array.
    data_ori : `array`
        The original data corresponding to `x_uni_ori`.
    bandwidth : `str`, optional
        The bandwidth to use for the kernel density estimation. Defaults to 'scott'.
    kernel_func : `str`, optional
        The kernel function to use for KDE. Defaults to 'epanechnikov'.
    
    Returns
    -------
    sklearn.neighbors.kde.KernelDensity
        The fitted KDE model.
    """    
    # KernelDensity requires input to be a column vector
    # So x_uni_re must be reshaped
    x_uni_ori_re = x_uni_ori.reshape(-1, 1)
    # Avoid divide-by-zero warnings when calculating KDE
    data_ori_adjested = np.where(data_ori <= 0, 1e-20, data_ori)      
    kde = KernelDensity(kernel=kernel_func, bandwidth=bandwidth)
    kde.fit(x_uni_ori_re, sample_weight=data_ori_adjested)  
    return kde

def KDE_score(self, kde, x_uni_new):
    """
    Evaluate the KDE model on new data points and normalize the results.
    
    Parameters
    ----------
    kde : sklearn.neighbors.kde.KernelDensity
        The fitted KDE model from :meth:`~.KDE_fit`.
    x_uni_new :`array`
        New unique data points where the KDE model is evaluated.
    
    Returns
    -------
    `array`
        The smoothed and normalized data based on the KDE model.
    """
    x_uni_new_re = x_uni_new.reshape(-1, 1) 
    data_smoothing = np.exp(kde.score_samples(x_uni_new_re))
    
    # Flatten a column vector into a one-dimensional array
    data_smoothing = data_smoothing.ravel()
    ## normalize the data for value range after smoothing
    # data_smoothing = data_smoothing/np.trapz(data_smoothing,x_uni_new)
    Q3 = self.calc_Q3(x_uni_new, data_smoothing)
    data_smoothing = data_smoothing / Q3[-1]
    return data_smoothing

def traverse_path(self, label, path_ori):
    """
    Update the file path or list of paths based on the label.
    
    Parameters
    ----------
    label : `int`
        The label to update in the file path(s).
    path_ori : `str` or `list` of `str`
        The original file path or list of file paths to be updated.
    
    Returns
    -------
    `str` or `list` of `str`
        The updated file path.
    """
    def update_path(path, label):
        if label == 0:
            return path.replace(".xlsx", f"_{label}.xlsx")
        else:
            return path.replace(f"_{label-1}.xlsx", f"_{label}.xlsx")

    if isinstance(path_ori, list):
        return [update_path(path, label) for path in path_ori]
    else:
        return update_path(path_ori, label)    
