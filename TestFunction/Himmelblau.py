# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:29:42 2023

@author: px2030
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping, minimize, brute

def calculate_model_function(XY, mod_fun = 'rastrigin', c1=20, c2=2, c3=2):
    # XY is a 2d array, where XY[0]=X and XY[1]=Y
    X = XY[0]
    Y = XY[1]
    
    if mod_fun == 'rastrigin':
        # Rastrigin function https://en.wikipedia.org/wiki/Test_functions_for_optimization
        Z = c1 + X**2 - c2 * np.cos(2*np.pi*X) + Y**2 - c3 * np.cos(2*np.pi*Y)
        
    elif mod_fun == 'himmelblau':
        Z = (X**2 + Y -11)**2 + (X + Y**2 - 7)**2
        
    return Z
    
def find_minimum(x = (-5,5), XY0 = [-5,-5], ax=None, opt_algo='basinhopping'):
    
    if opt_algo == 'basinhopping':
        minimizer_kwargs = {"method": "BFGS"}
        res = basinhopping(calculate_model_function, XY0, minimizer_kwargs=minimizer_kwargs,
                           niter=200)
    elif opt_algo == 'minimize':
        res = minimize(calculate_model_function, XY0, method='Nelder-Mead', tol=1e-6)
        
    elif opt_algo == 'brute':
        step = (x[-1] - x[0]) / (len(x) - 1)
        ranges = (slice(x[0], x[-1], step), slice(x[0], x[-1], step))
        res = brute(calculate_model_function, ranges, args=('himmelblau',), 
                    Ns=len(x), full_output=True, finish=minimize)
    
    else:
        print('No supported optimization algorithm provided. Returning None..')
        return None
    
    if opt_algo == 'brute':
        print(f"Global Minimum: [x,y] = {res[0]} with f([x,y]) = {res[1]}")
        if ax is not None:
            ax.scatter(XY0[0],XY0[1],calculate_model_function(XY0), s=200, color='b')
            ax.scatter(res[0][0],res[0][1],res[1], s=200, color='r')
            ax.set_title('Optimized with ' + opt_algo + '.')
    else:    
        print(f"Global Minimum: [x,y] = {res.x} with f([x,y]) = {res.fun}")
        
        if ax is not None:
            ax.scatter(XY0[0],XY0[1],calculate_model_function(XY0), s=200, color='b')
            ax.scatter(res.x[0],res.x[1],res.fun, s=200, color='r')
            ax.set_title('Optimized with ' + opt_algo + '.')
    return res
    
    
def plot_function(X, Y, Z, ax=None, fig=None):
    
    if ax is None or fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection='3d')
    
    ax.plot_surface(X, Y, Z)
    ax.view_init(elev=15, azim=-65)
    
    return ax, fig
    
if __name__ == '__main__':
    plt.close('all')
    
    x = np.linspace(-5,5,100)
    X, Y = np.meshgrid(x, x)
    
    Z = calculate_model_function([X, Y], 'himmelblau') 
    
    ax, fig = plot_function(X, Y, Z)
    
    # opt_algo = 'basinhopping'
    # opt_algo = 'minimize'
    opt_algo = 'brute'
    XY0 = [-4.23,-3.43]
    res = find_minimum(x = x, XY0 = XY0, ax = ax, opt_algo = opt_algo)
    
    plt.tight_layout()

    