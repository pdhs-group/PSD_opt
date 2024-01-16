# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 14:49:32 2023

@author: px2030
"""

import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
from functools import partial
from bayes_opt import BayesianOptimization

class opt_problem():

    def calculate_model_function(self, X, Y, mod_fun='rastrigin', c1=20, c2=2, c3=2, scale=1):
        if mod_fun == 'rastrigin':
            Z = c1 + X**2 - c2 * np.cos(2*np.pi*X) + Y**2 - c3 * np.cos(2*np.pi*Y)
        elif mod_fun == 'himmelblau':
            Z = (X**2 + Y -11)**2 + (X + Y**2 - 7)**2
        return Z*scale
    
    def bayes_opt(self):
        
        if self.opt_algo == 'gp_minimize':
            scale = 1
            objective = partial(self.calculate_model_function, scale=scale)
            opt = gp_minimize(
                lambda xy: objective(xy[0], xy[1]),
                [Real(-5.0, 5.0), Real(-5.0, 5.0)],
                n_calls=50,
                random_state=0
            )
        
            print(f"Global Minimum: [x,y] = {opt.x} with f([x,y]) = {opt.fun}")
            if self.ax is not None:
                self.ax.scatter(opt.x[0],opt.x[1],opt.fun, s=200, color='r')
                self.ax.set_title('Optimized with ' + self.opt_algo + '.')
                
        elif self.opt_algo == 'BayesianOptimization':
            scale = -1
            pbounds = {'X': (-5, 0), 'Y': (-5, 0)}
            objective = partial(self.calculate_model_function, mod_fun=self.mod_fun, scale=scale)
            opt = BayesianOptimization(
                f=objective, 
                pbounds=pbounds,
                random_state=1,
                allow_duplicate_points=True
            )
            
            opt.maximize(
                init_points=5,
                n_iter=100,
            )   
            
            print(f"Global Maximum: [x,y] = [{opt.max['params']['X']}, {opt.max['params']['Y']}] with f([x,y]) = {opt.max['target']}")
            if self.ax is not None:
                self.ax.scatter(opt.max['params']['X'], opt.max['params']['Y'], opt.max['target'], s=200, color='r')
                self.ax.set_title(f"Optimized with {self.opt_algo}.")
            
        return opt
        
    def plot_function(self, X, Y, Z, ax=None, fig=None):
        
        if ax is None or fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1, projection='3d')
        
        ax.plot_surface(X, Y, Z)
        ax.view_init(elev=15, azim=-65)
        
        return ax, fig
    
    def __init__(self, opt_algo='BayesianOptimization', mod_fun='himmelblau'):
        self.opt_algo = opt_algo
        self.mod_fun=mod_fun
        self.x = np.linspace(-5,5,100)
        self.X, self.Y = np.meshgrid(self.x, self.x)
        
        self.Z = self.calculate_model_function(self.X, self.Y, mod_fun=self.mod_fun) 
        
        self.ax, self.fig = self.plot_function(self.X, self.Y, self.Z)
        

    
    
    
if __name__ == '__main__':
    plt.close('all')
    
    problem = opt_problem(opt_algo='BayesianOptimization', mod_fun='himmelblau')
    problem.bayes_opt()
    
    plt.tight_layout()