# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:58:09 2023

@author: px2030
"""
import numpy as np
# import pandas
from generate_psd import full_psd
from pop import population
## For plots
import matplotlib.pyplot as plt
import plotter.plotter as pt   

if __name__ == '__main__':
    x50 = 1   # /mm
    resigma = 1
    minscale = 1e-3
    maxscale = 1e3
    dist_path = full_psd(x50, resigma, minscale=minscale, maxscale=maxscale, plot_psd=True)
    psd_dict = np.load(dist_path,allow_pickle=True).item()
    p = population(dim=2, disc='geo')
    p.DIST1 = dist_path
    p.DIST3 = dist_path
    p.R01 = psd_dict['r0_005']
    p.R03 = psd_dict['r0_005']
    p.full_init()
    p.solve_PBE()
    x_uni, q3, Q3, x_10, x_50, x_90 = p.return_num_distribution_fixed(t=10)
    pt.plot_init(scl_a4=1,figsze=[12.8,6.4*1.5],lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
    fig=plt.figure()    
    axq3=fig.add_subplot(1,2,1)   
    axQ3=fig.add_subplot(1,2,2)   
    
    axq3, fig = pt.plot_data(x_uni, q3, fig=fig, ax=axq3,
                           xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                           ylbl='number distribution of agglomerates $q3$ / $-$',
                           lbl='q3',clr='b',mrk='o')
    
    axQ3, fig = pt.plot_data(x_uni, Q3, fig=fig, ax=axQ3,
                           xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                           ylbl='accumulated number distribution of agglomerates $Q3$ / $-$',
                           lbl='Q3',clr='b',mrk='o')

    axq3.grid('minor')
    axQ3.grid('minor')
    plt.tight_layout() 