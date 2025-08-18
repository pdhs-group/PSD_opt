    ### ------ Static Methods ------ ### 
    
import numpy as np
from sklearn.neighbors import KernelDensity
## Interpolate PSD
def interpolate_psd(d,psd_data,v0,x_init=None,Q_init=None):
    """
    Obtain initial conditions from a PSD data file.
    
    Parameters
    ----------
    d : `array_like`
        Particle size grid on which the PSD should be interpolated. NOTE: This
        vector contains diameters
    psd_data : `str`
        Complete path (including filename) to datafile in which the PSD is saved. 
    v0: `float`
        Total VOLUME the distribution should be scaled to
    x_init : `array_like`, optional
        Particle size grid in the PSD can be manually specified.  
    Q_init : `array_like`, optional
        Manually specify the PSD instead of reading it from the file.
    Output
    ----------
    n : `array_like`
        NUMBER concentration vector corresponding to d
    """
    
    from scipy.interpolate import interp1d
    import sys
    
    ## OUTPUT-parameters:
    # n: NUMBER concentration vector corresponding to 2*r (for direct usage in N
    # vector of population balance calculation
    
    ## INPUT-parameters:
    # d: Particle size grid on which the PSD should be interpolated. NOTE: This
    #    vector contains diameters
    # PSD_data: Complete path (including filename) to datafile in which the PSD is saved. 
    #           This file should only contain 2 variables: Q_PSD and x_PSD.
    #           Here, x_PSD contains diameters (standard format of PSD)
    # v0: Total VOLUME the distribution should be scaled to
            
    
    # If x and Q are not directly given import from file
    if x_init is None and Q_init is None:
        # Import x values from dictionary save in psd_data
        psd_dict = np.load(psd_data,allow_pickle=True).item()
        x = psd_dict['x_PSD']
        
        ## Initializing the variables
        n = np.zeros(len(d)) 
        
        if 'Q_PSD' not in psd_dict and 'q_PSD' not in psd_dict:
            sys.exit("ERROR: Neither Q_PSD nor q_PSD given in distribution file. Exiting..")
        
        if 'Q_PSD' in psd_dict:
            # Load Q if given
            Q = psd_dict['Q_PSD']
                
    else:
        ## Initializing the variables
        n = np.zeros(len(d)) 
        x = x_init
        Q = Q_init
    
    # Interpolate Q on d grid and normalize it to 1 (account for numerical error)
    # If the ranges don't match well, insert 0. This is legit since it is the density
    # distribution
    f_Q_tem = interp1d(x,Q,bounds_error=False,fill_value=np.nan)
    f_Q = wrap_with_linear_extrapolation(f_Q_tem, x, Q)
    Q_d = np.zeros(len(d)+1)
    Q_d[1:] = np.clip(f_Q(d), 0.0, 1.0)
            
    #Q_d(math.isnan(Q_d)) = 0
    Q_d = Q_d/Q_d.max()
    
    for i in range(1, len(d)+1):
        v_total_tmp = max(v0 * (Q_d[i] - Q_d[i-1]), 0)
        v_one_tmp = (1/6)*np.pi*d[i-1]**3
        n[i-1] = v_total_tmp/v_one_tmp
    
    # # Eliminate sub and near zero values (sub-thrshold)
    # thr = 1e-5
    # n[n<thr*np.mean(n)] = 0
    
    return n    

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

## Plot 2D-distribution:
def plot_N2D(N,V,V0_tot,ax=None,fig=None,close_all=False,scl_a4=1,figsze=[12.8*1.05,12.8],THR_N=1e-4,
             exp_file=None,t_stamp=None):
                
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import plotter.plotter as pt
    from plotter.KIT_cmap import KIT_black_green_white
    
    if close_all:
        plt.close('all')
        
    if fig is None:
        pt.plot_init(scl_a4=scl_a4,figsze=figsze,lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
        fig = plt.figure()    
        ax = fig.add_subplot(1,1,1)
    
    # Clear axes
    ax.cla()
    
    # Calculate meshgrid for plot
    _ii, _jj = np.meshgrid(np.arange(len(N)),np.arange(len(N)))
                    
    # Calculate relative Volume and apply threshold
    Nr = 100*(N*V)/V0_tot
    Nr[Nr<THR_N]=THR_N
    
    # Color plot
    cp = ax.pcolor(_ii,_jj,Nr,norm=LogNorm(vmin=1e3*THR_N, vmax=100),edgecolors=[0.5,0.5,0.5],
                   shading='auto',cmap=KIT_black_green_white.reversed()) 
    
    # Colorbar / legend
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = plt.colorbar(cp,cax)
    c = cb.set_label('Total volume $V$ / $\%$')
   
    # Format the plot
    ax.set_ylabel('Partial volume comp. 1 $V_{1}$ ($i$) / $-$')  # Add an x-label to the axes.
    ax.set_xlabel('Partial volume comp. 2 $V_{2}$ ($j$) / $-$')  # Add a y-label to the axes.
    
    # Plot time textbox if t_stamp is provided
    if t_stamp is not None:
        ax.text(0.05, 0.95, f"$t={t_stamp}$", transform=ax.transAxes, fontsize=10*1.6,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='w', alpha=1))
    
    #Remove whitespace around plot
    plt.tight_layout()
    
    # Plot frame
    if exp_file is not None: pt.plot_export(exp_file)
    
    return ax, cb, fig

def KDE_fit(x_uni_ori, data_ori, bandwidth='scott', kernel_func='epanechnikov'):
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

def KDE_score(kde, x_uni_new):
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
    Qx = np.zeros_like(data_smoothing)
    for i in range(1, len(Qx)):
        Qx[i] = Qx[i-1] + data_smoothing[i] * (x_uni_new[i] - x_uni_new[i-1])
    data_smoothing = data_smoothing / Qx[-1]
    
    return data_smoothing