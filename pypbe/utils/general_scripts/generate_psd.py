# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 09:03:36 2021

@author: xy0264
"""
import numpy as np
from scipy.stats import norm,lognorm

def generate_psd_normal(x50,sigma,exp_name=None,xmin=None,xmax=None):
    
    # Generate the particle size distributions Q (sum distribution) and q (density distribution)
    # over the diameter x, according to GAUSSIAN or NORMAL distribution.
    
    ## INPUT-parameters:
    # x50:   Median diameter of the distribution. Q(x50)=0.5. 
    # sigma: Standard deviation or variance of normal distribution.
    # xmin:  OPTIONAL, Minimum x value for returned distribution. Default: 1e-2*x50
    # xmax:  OPTIONAL, Maximum x value for returned distribution. Default: 1e2*x50
    
    ## OUTPUT-parameters:
    # Q_PSD: Sum distribution.
    # q_PSD: Densitiy distribution.
    # x_PSD: Particle diameter array.
    
    # Set xmin and xmax if not given by function:
    if xmin==None:
        xmin=x50*1e-2
    if xmax==None:
        xmax=x50*1e2
        
    # Number of x points in x_PSD
    NUM_X=10000
        
    # Generate logarithmically spaced x vector
    x_PSD=np.logspace(np.log10(xmin),np.log10(xmax),NUM_X)
    
    # Generate q_PSD according to normal distribution from scipy.stats module
    q_PSD=norm.pdf(x_PSD,x50,sigma)
    
    # Generate Q_PSD. It it the integral of q over x
    Q_PSD=np.zeros(np.shape(q_PSD))
    for i in range(1,len(Q_PSD)):
        Q_PSD[i]=np.trapz(q_PSD[:i+1],x_PSD[:i+1])
    Q_PSD = Q_PSD/Q_PSD[-1]    
    return Q_PSD, q_PSD, x_PSD 

def generate_psd_lognormal(x50,sigma,exp_name=None,xmin=None,xmax=None):
    if xmin is None:
        xmin = x50 * 1e-2
    if xmax is None:
        xmax = x50 * 1e2
        
    # Number of x points in x_PSD
    NUM_X = 10000
        
    # Generate logarithmically spaced x vector
    x_PSD = np.logspace(np.log10(xmin), np.log10(xmax), NUM_X)
    
    # Calculate scale parameter for lognorm function
    s = abs(np.log(sigma))
    
    # Generate q_PSD according to log-normal distribution
    q_PSD = lognorm.pdf(x_PSD, s, scale=x50)
    
    # Generate Q_PSD. It is the integral of q over x
    Q_PSD = np.zeros_like(q_PSD)
    for i in range(1, len(Q_PSD)):
        Q_PSD[i] = np.trapz(q_PSD[:i+1], x_PSD[:i+1])
    Q_PSD = Q_PSD/Q_PSD[-1]
        
    return Q_PSD, q_PSD, x_PSD
    
def find_x_f(Q_PSD,x_PSD,F):
    
    ## INPUT-Parameters:
    # Q_PSD: Sum distribution.
    # x_PSD: Particle diameter array.    
    # F:     Fraction of Q_PSD to return e[0,1]
    
    ## OUTPUT-Parameters:
    # x_F:   Diameter where Q_PSD(x_F)=F
    
    # Find x_F corresponding to the given fraction
    idx_F = (np.abs(Q_PSD - F)).argmin()
    x_F = x_PSD[idx_F]    
    
    return x_F

def full_psd(x50, resigma=0.2, minscale=None, maxscale=None, plot_psd=False, output_dir=None):
    
    ## INPUT-Parameters:
    # x50:   Median diameter of the distribution. Q(x50)=0.5. 
    # sigma: Standard deviation or variance of normal distribution.  
    # F:     Fraction of Q_PSD to return e[0,1]
    
    ## OUTPUT-Parameters:
    # r0:    Radius corresponting to Q_PSD(2*r0)=F
    # dist:  Full path to saved PSD file
    
    import os
    from decimal import Decimal
    scl=1e-6
    x50 *= scl 
    # Volume specific
    v50=np.pi*x50**3/6
    sigma=v50*resigma
    vmin=v50*minscale
    vmax=v50*maxscale
    Q,q,v = generate_psd_normal(v50,sigma,xmin=vmin,xmax=vmax)
    x = (6*v/np.pi)**(1/3)
    
    # sigma = x50*resigma
    # xmin=x50*minscale
    # xmax=x50*maxscale
    # Q,q,x = generate_psd_lognormal(x50,sigma,xmin=xmin,xmax=xmax)
    
    # Find r0 corresponding to the volume fraction of 1%, 5%, and 10%
    r0_001 = find_x_f(Q,x,0.01)/2
    r0_005 = find_x_f(Q,x,0.05)/2
    r0_01 = find_x_f(Q,x,0.1)/2
    
    if plot_psd == True:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        ax.plot(x,q/np.max(q))  # Plot some data on the axes.
        ax.plot(x,Q)
        plt.axvline(x=r0_005,color='k')
        ax.set_xscale('log')
        ax.legend(['Density distribution q','Sum distribution Q'])
    # Generate full3 filestring
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname( __file__ ),"..","..","data","PSD_data")
    os.makedirs(output_dir, exist_ok=True)
    dist = os.path.join(output_dir, f"PSD_x50_{Decimal(x50):.1E}_RelSigmaV_{Decimal(resigma):.1E}.npy")
    
    # Create and save PSD dictionary
    dict_Qx={'Q_PSD':Q,'x_PSD':x, 'r0_001':r0_001, 'r0_005':r0_005, 'r0_01':r0_01}
    np.save(dist,dict_Qx)  
       
    return dist
        
if __name__ == '__main__':
    x50 = 20  # /um
    resigma = 0.2
    minscale = 0.01
    maxscale = 100
    full_psd(x50, resigma, minscale=minscale, maxscale=maxscale, plot_psd=True)