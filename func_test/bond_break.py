# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:44:42 2024

@author: xy0264
"""
#from icecream import ic 
import numpy as np
import matplotlib.pyplot as plt
import plotter.plotter as pt          
from plotter.KIT_cmap import c_KIT_green, c_KIT_red, c_KIT_blue
pt.close()
pt.plot_init(mrksze=12,lnewdth=1)

def float_gcd(a, b, rtol = 1e-3, atol = 1e-8):
    t = min(abs(a), abs(b))
    while abs(b) > rtol * t + atol:
        a, b = b, a % b
    return a

def break_something_random(e_v, e_h):
    c = np.random.choice([0,1,2,3])
    if c == 0:
        # Break from left
        i_init = np.random.choice(np.arange(1,e_h.shape[0])) 
        j_init = 0
        e_h[i_init, j_init] = -1
    if c == 1:
        # Break from right
        i_init = np.random.choice(np.arange(1,e_h.shape[0])) 
        j_init = -1
        e_h[i_init, j_init] = -1
    if c == 2:
        # Break from top
        j_init = np.random.choice(np.arange(1,e_v.shape[0])) 
        i_init = 0
        e_v[i_init, j_init] = -1
    if c == 3:
        # Break from bottom
        j_init = np.random.choice(np.arange(1,e_v.shape[0])) 
        i_init = -1
        e_v[i_init, j_init] = -1
    
    print('Broke bond at index ', i_init, j_init)
    return e_v, e_h

    
# Generate 2D grids for pivots and edges
def generate_grids_2D(A, X1, X2):
    A0 = float_gcd(A*X1, A*X2)
    # N: array with total number of squares [1, 2] 
    N = np.array([int(A*X1/A0), int(A*X2/A0)])
    DIM = int(np.ceil(np.sqrt(np.sum(N))))
    
    # B: array with total number of bonds [11, 12, 22]
    B = np.zeros(3) 
    
    p = np.ones((DIM,DIM))*(-1)
    e_v = np.ones((DIM,DIM+1))*(-1)
    e_h = np.ones((DIM+1,DIM))*(-1)
    
    # temporary counter
    N1 = N[0]
    N2 = N[1]
    for i in range(DIM):
        for j in range(DIM):
            # Select 1 or 2 for p
            if N1>0 and N2>0:
                p[i,j] = np.random.choice([1,2], p=[X1,X2])
            elif N1==0 and N2>0:
                p[i,j] = 2 
            elif N1>0 and N2==0:
                p[i,j] = 1
                
            if i>0:
                e_h[i,j]=p[i,j]*p[i-1,j]
            if j>0:
                e_v[i,j]=p[i,j]*p[i,j-1]
            
            # Adjust counter
            N1 -= int(p[i,j]==1)
            N2 -= int(p[i,j]==2)
            
            B += np.array([int(e_h[i,j]==1),int(e_h[i,j]==2),int(e_h[i,j]==4)]) 
            B += np.array([int(e_v[i,j]==1),int(e_v[i,j]==2),int(e_v[i,j]==4)])                      
            
    return p, e_v, e_h, N, B, A0

if __name__ == '__main__':
    
    A = 10
    X1 = 0.23
    X2 = 1-X1
    
    # Generate grid
    p, e_v, e_h, N, B, A0 = generate_grids_2D(A, X1, X2)
    #print(A0, A*X1/A0, A*X2/A0)
    
    # Break a single bordering bond
    e_v, e_h = break_something_random(e_v, e_h)
    
    i1, j1 = np.where(p==1)
    i2, j2 = np.where(p==2)
    
    iev11, jev11 = np.where(e_v==1)
    iev12, jev12 = np.where(e_v==2)    
    iev22, jev22 = np.where(e_v==4) 
    iev_b, jev_b = np.where(e_v==(-1))
    
    ieh11, jeh11 = np.where(e_h==1)
    ieh12, jeh12 = np.where(e_h==2)    
    ieh22, jeh22 = np.where(e_h==4)
    ieh_b, jeh_b = np.where(e_h==(-1))
       
    fig=plt.figure(figsize=[5,5])    
    ax=fig.add_subplot(1,1,1) 
    
    pt.plot_init(mrksze=16,lnewdth=1)
    ax.scatter(j1,i1, marker='s', color=c_KIT_green, label='1')
    ax.scatter(j2,i2, marker='s', color=c_KIT_red, label='2')
    pt.plot_init(mrksze=8,lnewdth=1)
    ax.scatter(jev11-0.5, iev11, marker='^', color=c_KIT_green, label='11')
    ax.scatter(jev12-0.5, iev12, marker='^', color=c_KIT_blue, label='12')
    ax.scatter(jev22-0.5, iev22, marker='^', color=c_KIT_red, label='22')
    ax.scatter(jev_b-0.5, iev_b, marker='^', color='k', label='edge')
    ax.scatter(jeh11, ieh11-0.5, marker='^', color=c_KIT_green)
    ax.scatter(jeh12, ieh12-0.5, marker='^', color=c_KIT_blue)
    ax.scatter(jeh22, ieh22-0.5, marker='^', color=c_KIT_red)
    ax.scatter(jeh_b, ieh_b-0.5, marker='^', color='k')
    
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    
