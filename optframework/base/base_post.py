# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 15:29:51 2025

@author: px2030
"""

import numpy as np
import matplotlib.pyplot as plt 
from optframework.utils.plotter import plotter as pt  
from optframework.utils.plotter.KIT_cmap import c_KIT_green, c_KIT_red, c_KIT_blue

class BasePost():
    def _visualize_mom_t(self, mu, mu_std=None, i=0, j=0, t_vec=None, 
                         fig=None, ax=None, clr=c_KIT_green, lbl='MC'):
        if fig is None or ax is None:
            fig=plt.figure()    
            ax=fig.add_subplot(1,1,1)   
        
        ax, fig = pt.plot_data(t_vec,mu[i,j,:], err=mu_std[i,j,:], fig=fig, ax=ax,
                               xlbl='Process time $t_\mathrm{A}$ / $s$',
                               ylbl=f'Moment $\mu ({i}{j})$ / '+'$m^{3\cdot'+str(i+j)+'}$',
                               lbl=lbl,clr=clr,mrk='o')
                
        # Adjust y scale in case of first moment
        if i+j == 1:
            ax.set_ylim([np.min(mu[i,j,:])*0.9,np.max(mu[i,j,:])*1.1])
        ax.grid('minor')
        plt.tight_layout()   
        
        return ax, fig 
        