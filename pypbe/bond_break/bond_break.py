# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:44:42 2024

@author: xy0264
"""
#from icecream import ic 
import sys
import os
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),"../.."))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import pypbe.utils.plotter.plotter as pt        
from pypbe.utils.plotter.KIT_cmap import c_KIT_green, c_KIT_red, c_KIT_blue, KIT_black_green_white
pt.close()
pt.plot_init(mrksze=12,lnewdth=1)
from copy import deepcopy
from numba import jit

def float_gcd(a, b, rtol = 1e-3, atol = 1e-8):
    t = min(abs(a), abs(b))
    while abs(b) > rtol * t + atol:
        a, b = b, a % b
    return a

# Generate 2D grid containing both pivots and edges
def generate_grid_2D(A, X1, X2, A0=None):
    if A0 is None or A0>(A*X1/2) or A0>(A*X2/2):
        A0 = float_gcd(A*X1, A*X2)

    # N: array with total number of squares [1, 2]
    # R: array with rest area [1, 2]
    N = np.array([int((A*X1)//A0), int((A*X2)//A0)]) 
    R = np.array([(A*X1)%A0, (A*X2)%A0])
    DIM = int(np.ceil(np.sqrt(np.sum(N))))
    
    # B: array with total number of bonds [11, 12, 22]
    B = np.zeros(3) 
    
    # G: Grid containing both pivots (squares), bonds and edges
    # pivots: 1 or 2 (material 1 or 2)
    # bonds: -1, 11, 12 or 22 (combinations of 1 and 2)
    #           -1 is an "outside" surface with no contact
    # edges: 0
    G = np.ones((2*DIM+1,2*DIM+1))*(-1)
    
    # Set edges (starting a [0,0] in steps of 2)
    for i in range(0,2*DIM+1,2):
        for j in range(0,2*DIM+1,2):
            G[i,j] = 0
    
    # temporary counter
    N1 = N[0]
    N2 = N[1]
    # Loop over all PIVOTS. They start at [1,1] and go in index steps of 2 (bond in between)
    for i in range(1,2*DIM,2):
        for j in range(1,2*DIM,2):
            # Set pivot
            if N1>0 and N2>0:
                G[i,j] = np.random.choice([1,2], p=[X1,X2])
            elif N1==0 and N2>0:
                G[i,j] = 2 
            elif N1>0 and N2==0:
                G[i,j] = 1
                        
            # Adjust counter
            N1 -= int(G[i,j]==1)
            N2 -= int(G[i,j]==2)
            
            # Set bonds and update counter
            if i>1:
                if G[i-2,j] == 1:
                    if G[i,j] == 1:
                        G[i-1,j] = 11
                        B[0] += 1
                    elif G[i,j] == 2:
                        G[i-1,j] = 12
                        B[1] += 1
                elif G[i-2,j] == 2:
                    if G[i,j] == 1:
                        G[i-1,j] = 12
                        B[1] += 1
                    elif G[i,j] == 2:
                        G[i-1,j] = 22
                        B[2] += 1

            if j>1:
                if G[i,j-2] == 1:
                    if G[i,j] == 1:
                        G[i,j-1] = 11
                        B[0] += 1
                    elif G[i,j] == 2:
                        G[i,j-1] = 12
                        B[1] += 1
                elif G[i,j-2] == 2:
                    if G[i,j] == 1:
                        G[i,j-1] = 12
                        B[1] += 1
                    elif G[i,j] == 2:
                        G[i,j-1] = 22
                        B[2] += 1                  
            
    return G, N, B, A0, R

def plot_G(G, title=None, fill_no=[], fill_clr=[]):
    i_p1, j_p1 = np.where(G==1)
    i_p2, j_p2 = np.where(G==2)
    
    i_b11, j_b11 = np.where(G==11)
    i_b12, j_b12 = np.where(G==12)  
    i_b22, j_b22 = np.where(G==22) 
    i_bm1, j_bm1 = np.where(G==-1)
    
    i_e, j_e = np.where(G==0)
    
    fig=plt.figure(figsize=[5,5])    
    ax=fig.add_subplot(1,1,1) 
    
    pt.plot_init(mrksze=16,lnewdth=1)
    ax.scatter(j_p1,i_p1, marker='s', color=c_KIT_green, label='1')
    ax.scatter(j_p2,i_p2, marker='s', color=c_KIT_red, label='2')
    pt.plot_init(mrksze=8,lnewdth=1)
    ax.scatter(j_b11, i_b11, marker='^', color=c_KIT_green, label='11')
    ax.scatter(j_b12, i_b12, marker='^', color=c_KIT_blue, label='12')
    ax.scatter(j_b22, i_b22, marker='^', color=c_KIT_red, label='22')
    ax.scatter(j_bm1, i_bm1, marker='^', color='k', label='no contact')
    pt.plot_init(mrksze=4,lnewdth=1)
    ax.scatter(j_e, i_e, marker='.', color='k', label='edge')
    
    for n in range(len(fill_no)):
        i_pf, j_pf = np.where(G==fill_no[n])
        pt.plot_init(mrksze=16,lnewdth=1)
        ax.scatter(j_pf,i_pf, marker='s', color=fill_clr[n], label=f'fill {n}')
        
    # ax.legend()
    ax.grid(True)
    ax.set_title(title)
    plt.tight_layout()
    
    return ax, fig

def break_one_bond(G, STR, idx=None, init_break_random=False):
    # STR: Array containing the strength of bonds [11,12,22]
    # idx: np.array([i,j]) indicating the index of current edge to propagate breakge
    #      None indicates that we start a new rupture (from the outside edge)
    
    G_new = np.copy(G)
    
    # Find initial edge to start rupture if idx=None
    # A valid edge must only have one or two neighboring "real" bonds (not -1)
    if idx is None:
        if init_break_random:
            flag = True
            while flag:
                # Generate random edge
                idx = np.random.choice(np.arange(0,G.shape[0],2), size=2)
                
                # Counter for valid bonds
                cnt = 0
                
                # Check surrounding bonds
                if idx[0] > 0:
                    if G[idx[0]-1, idx[1]] != -1:
                        cnt +=1 
                if idx[0] < G.shape[0]-1:
                    if G[idx[0]+1, idx[1]] != -1:
                        cnt +=1 
                if idx[1] > 0:
                    if G[idx[0], idx[1]-1] != -1:
                        cnt +=1 
                if idx[1] < G.shape[0]-1:
                    if G[idx[0], idx[1]+1] != -1:
                        cnt +=1
                
                if cnt == 1 or cnt == 2:
                    flag = False
        else:
            # List all breakable edges, their index and their corresponding bond strength
            # init_array[Number, TYPE, i, j, PROB]
            init_array = np.zeros((int(4*(G.shape[0]-3)/2),5))
            cnt = 0
            # Bottom and top row
            for i in range(2,G.shape[0]-1,2):
                init_array[cnt, :-1] = np.array([cnt, G[i,1], i, 0])                    
                init_array[cnt+1, :-1] = np.array([cnt+1, G[i,-2], i, G.shape[0]-1])
                cnt += 2
            # Left and right column
            for j in range(2,G.shape[0]-1,2):
                init_array[cnt, :-1] = np.array([cnt, G[1,j], 0, j])                    
                init_array[cnt+1, :-1] = np.array([cnt+1, G[-2,j], G.shape[0]-1, j])
                cnt += 2
            # Set probability column
            init_array[:,4][init_array[:,1]==11] = 1/STR[0]
            init_array[:,4][init_array[:,1]==12] = 1/STR[1]
            init_array[:,4][init_array[:,1]==22] = 1/STR[2]
            init_array[:,4] /= np.sum(init_array[:,4])
            
            b_idx = int(np.random.choice(init_array[:,0], p=init_array[:,4]))
            idx = np.array([init_array[b_idx,2],init_array[b_idx,3]]).astype(int)   
            #print('initial bond is type: ', init_array[b_idx,1])             
            
    # Start at idx and list all surrounding bonds that are breakable (not -1)
    b = np.zeros(4)
    if idx[0] > 0:
        b[0] = G[idx[0]-1,idx[1]]
    if idx[0] < G.shape[0]-1:
        b[1] = G[idx[0]+1,idx[1]] 
    if idx[1] > 0:
        b[2] = G[idx[0],idx[1]-1]
    if idx[1] < G.shape[0]-1:
        b[3] = G[idx[0],idx[1]+1]
        
    # Create probability and strength array for each bond 
    p = np.zeros(4)
    str_array = np.zeros(4)
    # p[b==-1] = 0 Not needed since initialized with 0
    p[b==11] = 1/STR[0]
    p[b==12] = 1/STR[1]
    p[b==22] = 1/STR[2]
    str_array[p!=0] = 1/p[p!=0]
        
    # Normalize probabilites
    p /= np.sum(p)
    
    # Select a bond to break
    b_idx = np.random.choice(np.arange(4), p=p)     
    
    # Progress the fracture
    if b_idx == 0:
        G_new[idx[0]-1,idx[1]] = -1
        idx_new = np.array([idx[0]-2,idx[1]]) 
        # idx[0] -= 2
    if b_idx == 1:
        G_new[idx[0]+1,idx[1]] = -1
        idx_new = np.array([idx[0]+2,idx[1]])
        # idx[0] += 2
    if b_idx == 2:
        G_new[idx[0],idx[1]-1] = -1
        idx_new = np.array([idx[0],idx[1]-2])
        # idx[1] -= 2
    if b_idx == 3:
        G_new[idx[0],idx[1]+1] = -1
        idx_new = np.array([idx[0],idx[1]+2])
        # idx[1] += 2
        
    # Check if this leads to a complete fracture (new index has not more than 1 breakable bonds)
    # Counter for valid bonds
    cnt = 0
    
    # Check surrounding bonds
    if idx_new[0] > 0:
        if G_new[idx_new[0]-1, idx_new[1]] != -1:
            cnt +=1 
    if idx_new[0] < G.shape[0]-1:
        if G_new[idx_new[0]+1, idx_new[1]] != -1:
            cnt +=1 
    if idx_new[1] > 0:
        if G_new[idx_new[0], idx_new[1]-1] != -1:
            cnt +=1 
    if idx_new[1] < G.shape[0]-1:
        if G_new[idx_new[0], idx_new[1]+1] != -1:
            cnt +=1
    
    if cnt>1:
        fracture_flag = False
    else:
        fracture_flag = True
        
    return G_new, idx_new, fracture_flag, str_array[b_idx]    

def recursive_fun(G, i, j, cnt_1, cnt_2, new_value=0):
    DIM = G.shape[0]
    
    if i<0 or i>=DIM or j<0 or j>=DIM or (G[i,j]!=1 and G[i,j]!=2):
        pass
    else:
        if G[i,j] == 1:
            cnt_1 += 1 
        elif G[i,j] == 2:
            cnt_2 += 1
        G[i,j] = new_value
        
        if G[i+1,j] != -1:
            G, cnt_1, cnt_2 = recursive_fun(G, i+2, j, cnt_1, cnt_2, new_value=new_value)
            G[i+1,j] = -2
        if G[i-1,j] != -1:
            G, cnt_1, cnt_2 = recursive_fun(G, i-2, j, cnt_1, cnt_2, new_value=new_value)
            G[i-1,j] = -2
        if G[i,j+1] != -1:
            G, cnt_1, cnt_2 = recursive_fun(G, i, j+2, cnt_1, cnt_2, new_value=new_value)
            G[i,j+1] = -2
        if G[i,j-1] != -1:
            G, cnt_1, cnt_2 = recursive_fun(G, i, j-2, cnt_1, cnt_2, new_value=new_value)
            G[i,j-1] = -2
    
    return G, cnt_1, cnt_2        

def iterative_fun(G, i, j, new_value=0):
    DIM = G.shape[0]
    stack = [(i, j)]
    cnt_1 = cnt_2 = 0
    
    while stack:
        ci, cj = stack.pop()
        if ci < 0 or ci >= DIM or cj < 0 or cj >= DIM or G[ci, cj] not in (1, 2):
            continue
    
        if G[ci, cj] == 1:
            cnt_1 += 1
        elif G[ci, cj] == 2:
            cnt_2 += 1
        
        G[ci, cj] = new_value  # Mark the cell as processed by changing its value
    
        # Add the adjacent material cells in the grid to the stack, checking connections first
        for di, dj in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
            conn_i, conn_j = ci + di//2, cj + dj//2  # Connection index
            ni, nj = ci + di, cj + dj  # Next material cell index
            
            if 0 <= ni < DIM and 0 <= nj < DIM:
                if G[conn_i, conn_j] not in (-1, -2):  # Check if connection is not broken
                    stack.append((ni, nj))
                    G[conn_i, conn_j] = -2  # Mark the connection as checked
    
    return G, cnt_1, cnt_2
    
def analyze_fragments(G):
    DIM = G.shape[0]
    
    cnt_1_arr = []
    cnt_2_arr = []
    val_arr = []
    val_cnt = 0
    for i in range(1,DIM,2):
        for j in range(1,DIM,2):
            if G[i,j]==1 or G[i,j]==2:
                # G, cnt_1, cnt_2 = recursive_fun(G, i, j, 0, 0, new_value=3+val_cnt) 
                G, cnt_1, cnt_2 = iterative_fun(G, i, j, new_value=3+val_cnt) 
                cnt_1_arr.append(cnt_1)
                cnt_2_arr.append(cnt_2)
                val_arr.append(3+val_cnt)
                val_cnt += 1
                
    return G, np.array(cnt_1_arr), np.array(cnt_2_arr), val_arr 
              
def check_idx_hist(idx, idx_hist):
    return np.any(np.all(idx == idx_hist, axis=1))  

def check_deadend(idx, idx_hist, G):
    # A deadend is reached (return True) if 
    #     all surrounding pivots are in idx_hist
    #     or surrounding bonds are -1 (no contact) 
    # bool_list = [np.any(np.all(idx+[2,0] == idx_hist, axis=1)) or G[tuple(idx+[1,0])]==-1,
    #              np.any(np.all(idx-[2,0] == idx_hist, axis=1)) or G[tuple(idx-[1,0])]==-1,
    #              np.any(np.all(idx+[0,2] == idx_hist, axis=1)) or G[tuple(idx+[0,1])]==-1,
    #              np.any(np.all(idx-[0,2] == idx_hist, axis=1)) or G[tuple(idx-[0,1])]==-1]
    bool_list = [np.any(np.all(idx+[2,0] == idx_hist, axis=1)),
                 np.any(np.all(idx-[2,0] == idx_hist, axis=1)),
                 np.any(np.all(idx+[0,2] == idx_hist, axis=1)),
                 np.any(np.all(idx-[0,2] == idx_hist, axis=1))]
    
    return all(bool_list)

def single_sim(A, X1, X2, STR, NO_FRAG, A0=None, init_break_random=False, plot=True, 
               close=False, verbose=True):
    
    if close: plt.close('all')
    
    # Generate and plot grid
    G, N, B, A0, R = generate_grid_2D(A, X1, X2, A0=A0)
    G0 = np.copy(G)
    
    # print(A0, A*X1/A0, A*X2/A0)
    if plot: ax0, fig0 = plot_G(G0, title='Initial grid')
    
    # Breaking stuff
    # Tracking number of fragments and index history of all fragments
    no_frag = 1
    idx_hist_frag = []
    fracture_energy = 0

    while no_frag < NO_FRAG:
        # For each fracture keep a separate history (otherwise fragments cannot break "inside" themselves)
        idx_hist = []
        
        if verbose: print(f'Starting fracture. Currently at {no_frag} fragments')
        # Initialize a new fracture. idx=None indicates that this is the first event
        G, idx, ff, str_bond = break_one_bond(G, STR, idx=None, init_break_random=init_break_random)
        idx_hist.append(np.copy(idx))
        fracture_energy += str_bond
        
        # Pursue this fracture until it breaks through 
        # Rare cases lead to an endless loop (despite check_deadend call)
        # In this case simply repeat the fracture process from the beginning!
        cnt = 0
        while ff is False and cnt < 2*G.shape[0]:
            G_tmp, idx_tmp, ff_tmp, str_bond = break_one_bond(G, STR, idx=idx)
            
            # Final Fracture is always valid
            # Check for circular fracture (if not so, keep the result)
            # Also check surrounding nodes for circular fracture (endless loop otherwise / deadend)
            if not check_idx_hist(idx_tmp, idx_hist) \
                and not check_deadend(idx_tmp, idx_hist, G): 
                
                idx = np.copy(idx_tmp)
                G = np.copy(G_tmp)
                ff = ff_tmp
                idx_hist.append(np.copy(idx_tmp))
                fracture_energy += str_bond
                # print(f'valid event')
            else:
                if verbose: print(f'index {idx_tmp} already inside idx_hist')
            cnt += 1
        
        # Caught in an endless loop. Report and restart the fragmentation (reset no_frag and idx_hist_frag)
        if cnt >= 2*G.shape[0]: 
            no_frag = 1
            idx_hist_frag = []
            G = np.copy(G0)            
            if verbose: print('Caught in an endless loop :( Restarting this fragmentation process..')
        else:
            # Increase number of fragments and append to overall history
            no_frag += 1
            idx_hist_frag += idx_hist
            
            # Plot current fracture
            if plot: _, _ = plot_G(G, title=f'Currently {no_frag} fragments')  
    
    # Analyze framents (use copy to retain original G)
    G_new = np.copy(G)
    G_new, cnt_1_arr, cnt_2_arr, val_arr = analyze_fragments(G_new)
    
    if plot: 
        # Generate random colors for filling
        colormap = plt.get_cmap('nipy_spectral')
        indices = np.random.randint(0, 256, size=len(val_arr))
        fill_clr = [to_rgba(colormap(i)) for i in indices]
        
        plot_G(G_new, fill_no=val_arr, fill_clr=fill_clr)
    
    # Corresponding F array
    F = np.zeros((NO_FRAG,4))
    
    X_F = (cnt_1_arr+cnt_2_arr)*A0/(A-np.sum(R))
    # Total area of each fragment
    F[:,0] = (cnt_1_arr+cnt_2_arr)*A0 + X_F*(R[0]+R[1])     
    # Partial area of component 1
    F[:,1] = (A0*cnt_1_arr + X_F*R[0]) / F[:,0]  
    # Partial area of component 2
    F[:,2] = 1 - F[:, 1]
    # Scale fracture energy depending on individual bond length
    # TO-DO: Physical thoughts required here
    F[:,3] = np.ones(NO_FRAG)*fracture_energy*np.sqrt(A0)
    
    return G, G_new, R, cnt_1_arr, cnt_2_arr, val_arr, fracture_energy, F
        
# Simulate N_GRIDS grids that each fracture N_FRACS times
# For debugging/testing use single_sim, as this function is "optimized" by not plotting and printing stuff
# TO-DO: Implement numba JIT with nopython=True (probably have to adjust all sub-functions)
def MC_breakage(A, X1, X2, STR, NO_FRAG, N_GRIDS=100, N_FRACS=100, A0=None, init_break_random=False):
    
    # Initialize fracture array (return)
    F = np.zeros((N_GRIDS*N_FRACS*NO_FRAG,4))
    
    # Loop through all grids based on A, X1 and X2
    for g in range(N_GRIDS):
        print(f'Calculating grid no. {g+1}/{N_GRIDS}')
        # Generate grid and copy it (identical initial conditions for other fractures)
        G, N, B, A0, R = generate_grid_2D(A, X1, X2, A0=A0)
        G0 = np.copy(G)
        
        for f in range(N_FRACS):
            # Tracking number of fragments and index history of all fragments
            no_frag = 1
            G = np.copy(G0)
            fracture_energy = 0
            
            while no_frag < NO_FRAG:
                # For each fracture keep a separate history (otherwise fragments cannot break "inside" themselves)
                idx_hist = []
                
                # Initialize a new fracture. idx=None indicates that this is the first event
                G, idx, ff, str_bond = break_one_bond(G, STR, idx=None, init_break_random=init_break_random)
                idx_hist.append(np.copy(idx))
                fracture_energy += str_bond
                
                # Pursue this fracture until it breaks through 
                # Rare cases lead to an endless loop (despite check_deadend call)
                # In this case simply repeat the fracture process from the beginning!
                cnt = 0
                while ff is False and cnt < 2*G.shape[0]:
                    G_tmp, idx_tmp, ff_tmp, str_bond = break_one_bond(G, STR, idx=idx)
                    
                    # Final Fracture is always valid
                    # Check for circular fracture (if not so, keep the result)
                    # Also check surrounding nodes for circular fracture (endless loop otherwise / deadend)
                    if not check_idx_hist(idx_tmp, idx_hist) \
                        and not check_deadend(idx_tmp, idx_hist, G): 
                        
                        idx = np.copy(idx_tmp)
                        G = np.copy(G_tmp)
                        ff = ff_tmp
                        idx_hist.append(np.copy(idx_tmp))
                        fracture_energy += str_bond
                    cnt += 1
                
                # Caught in an endless loop. Report and restart the fragmentation (reset no_frag and idx_hist_frag)
                if cnt >= 2*G.shape[0]: 
                    no_frag = 1
                    G = np.copy(G0)            
                else:
                    # Increase number of fragments and append to overall history
                    no_frag += 1
            
            # Analyze framents 
            G, cnt_1_tmp, cnt_2_tmp, val_arr = analyze_fragments(G)
            
            # Save fragment array F = [total area, X1, X2, fracture energy]
            # Adjust for the remainder of the material (mass conservation)
            idx_F = g*N_FRACS*NO_FRAG+f*NO_FRAG
            X_F = (cnt_1_tmp+cnt_2_tmp)*A0/(A-np.sum(R))
            # Total area of each fragment
            F[idx_F:idx_F+NO_FRAG, 0] = (cnt_1_tmp+cnt_2_tmp)*A0 + X_F*(R[0]+R[1])     
            # Partial area of component 1
            F[idx_F:idx_F+NO_FRAG, 1] = (A0*cnt_1_tmp + X_F*R[0]) / F[idx_F:idx_F+NO_FRAG, 0]  
            # Partial area of component 2
            F[idx_F:idx_F+NO_FRAG, 2] = 1 - F[idx_F:idx_F+NO_FRAG, 1]
            # Scale fracture energy depending on individual bond length
            # TO-DO: Physical thoughts required here
            F[idx_F:idx_F+NO_FRAG, 3] = np.ones(NO_FRAG)*fracture_energy*np.sqrt(A0)
                
    return F
    
# %% MAIN    
if __name__ == '__main__':
    ########### -----------
    import cProfile
    import pstats
    profiler = cProfile.Profile()
    profiler.enable()
    ########### -----------
    
    A = 4**6
    A0 = 1
    X1 = 0.21
    X2 = 1-X1
    STR = np.array([1,1,1])
    NO_FRAG = 4
    INIT_BREAK_RANDOM = False
    
    N_GRIDS, N_FRACS = 50, 50
    
    # Perform a single simulation (1 grid, 1 fracture) for visualization
    # G, G_new, R, cnt_1_arr, cnt_2_arr, val_arr, fracture_energy, F = \
    #     single_sim(A, X1, X2, STR, NO_FRAG, plot=True, A0=A0,
    #                close=True, init_break_random=INIT_BREAK_RANDOM)
    
    # Perform stochastic simulation
    
    # Fragment array [total area, X1, X2, fracture energy]
    F = MC_breakage(A, X1, X2, STR, NO_FRAG, N_GRIDS=N_GRIDS, N_FRACS=N_FRACS, 
                    A0=A0, init_break_random=INIT_BREAK_RANDOM)
    
    ########### -----------
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    print('#### Profiler of MC_breakage')
    stats.print_stats('MC_breakage')
    ########### -----------
    
    # %% PLOTS
    pt.close()
    # 2D Fragment distribution
    ax1, fig1, cb1, H1, xe1, ye1 = pt.plot_2d_hist(x=F[:,0]*F[:,1],y=F[:,0]*F[:,2],bins=(20,20),w=None,
                                                   scale=('lin','lin'), clr=KIT_black_green_white.reversed(),grd=True,
                                                   xlbl='Partial Volume 1 $V_1$ / $\mathrm{m^3}$', norm=False,
                                                   ylbl='Partial Volume 2 $V_2$ / $\mathrm{m^3}$', 
                                                   scale_hist='log', hist_thr=1e-4)
    
    
    # 1D Histogram of fracture energy   
    ax2, fig2, H2, xe2 = pt.plot_1d_hist(x=F[:,3],bins=100,scale='lin',xlbl='Fracture Energy / a.u.',
                                         ylbl='Counts / $-$',clr=c_KIT_green,norm=False, alpha=0.7)
    #ax2.set_yscale('log')
    
    # 2D Histogram of fracture energy vs. fragment size
    ax3, fig3, cb3, H3, xe3, ye3 = pt.plot_2d_hist(x=F[:,0],y=F[:,3],bins=(20,20),w=None,
                                                   scale=('lin','lin'), clr=KIT_black_green_white.reversed(),grd=True,
                                                   xlbl='Fragment Size $V$ / $\mathrm{m^3}$', norm=False,
                                                   ylbl='Fracture Energy / a.u.', 
                                                   scale_hist='log', hist_thr=1e-4)

    # ########### -----------
    # stats = pstats.Stats(profiler)
    # stats.sort_stats('cumulative')
    # stats.print_stats(20)
    # ########### -----------

    
    # OLD calculation without mass conservation (rest)
    
    #         # Append to solution array
    #         cnt_1_arr += list(cnt_1_tmp)
    #         cnt_2_arr += list(cnt_2_tmp)
    #         fracture_energy_arr += [fracture_energy for i in range(len(cnt_1_tmp))]
    # # Fragment array [total area, X1, X2, fracture energy]
    # F = np.zeros((len(cnt_1_arr),4))
    # F[:,0] = (np.array(cnt_1_arr) + np.array(cnt_2_arr))*A0    
    # F[:,1] = np.array(cnt_1_arr)/(np.array(cnt_1_arr)+np.array(cnt_2_arr))    
    # F[:,2] = np.array(cnt_2_arr)/(np.array(cnt_1_arr)+np.array(cnt_2_arr))
    # # Scale fracture energy depending on individual bond length
    # # TO-DO: Physical thoughts required here
    # F[:,3] = np.array(fracture_energy_arr)*np.sqrt(A0)
    
    # cb.formatter.set_powerlimits((0, 0))
    # cb.ax.yaxis.set_offset_position('left')
    
    # # Generate and plot grid
    # G, N, B, A0, R = generate_grid_2D(A, X1, X2)
    # # print(A0, A*X1/A0, A*X2/A0)
    # ax0, fig0 = plot_G(G, title='Initial grid')
    
    # # Breaking stuff
    # # Tracking number of fragments and index history of all fragments
    # no_frag = 1
    # idx_hist_frag = []

    # while no_frag < NO_FRAG:
    #     # For each fracture keep a separate history (otherwise fragments cannot break "inside" themselves)
    #     idx_hist = []
        
    #     print(f'Starting fracture. Currently at {no_frag} fragments')
    #     # Initialize a new fracture. idx=None indicates that this is the first event
    #     G, idx, ff = break_one_bond(G, STR, idx=None)
    #     idx_hist.append(np.copy(idx))
        
    #     # Pursue this fracture until it breaks through
    #     # cnt = 0
    #     while ff is False:# and cnt<100:
    #         print(idx)
    #         G_tmp, idx_tmp, ff_tmp = break_one_bond(G, STR, idx=idx)
            
    #         # Final Fracture is always valid
    #         # Check for circular fracture (if not so, keep the result)
    #         # Also check surrounding nodes for circular fracture (endless loop otherwise / deadend)
    #         if not check_idx_hist(idx_tmp, idx_hist) \
    #             and not check_deadend(idx_tmp, idx_hist, G): 
                
    #             idx = deepcopy(idx_tmp)
    #             G = deepcopy(G_tmp)
    #             ff = ff_tmp
    #             idx_hist.append(np.copy(idx_tmp))
    #             # print(f'valid event')
    #         else:
    #             print(f'index {idx_tmp} already inside idx_hist')
    #         # cnt += 1
        
    #     # Increase number of fragments and append to overall history
    #     no_frag += 1
    #     idx_hist_frag += idx_hist
        
    #     # Plot current fracture
    #     _, _ = plot_G(G, title=f'Currently {no_frag} fragments')  
        
    #     G_new = np.copy(G)
    #     # G_new, cnt_1, cnt_2 = recursive_fun(G_new, 1, 1, 0, 0, new_value=5)
    #     G_new, cnt_1_arr, cnt_2_arr, val_arr = analyze_fragments(G_new)
        
    #     # Generate random colors for filling
    #     colormap = plt.get_cmap('nipy_spectral')
    #     indices = np.random.randint(0, 256, size=len(val_arr))
    #     fill_clr = [to_rgba(colormap(i)) for i in indices]
        
    #     plot_G(G_new, fill_no=val_arr, fill_clr=fill_clr)
    
    # ARCHIEVE 
        
    # p, e_v, e_h, N, B, A0 = generate_grids_2D(A, X1, X2)
    
    # i1, j1 = np.where(p==1)
    # i2, j2 = np.where(p==2)
    
    # iev11, jev11 = np.where(e_v==1)
    # iev12, jev12 = np.where(e_v==2)    
    # iev22, jev22 = np.where(e_v==4) 
    # iev_b, jev_b = np.where(e_v==(-1))
    
    # ieh11, jeh11 = np.where(e_h==1)
    # ieh12, jeh12 = np.where(e_h==2)    
    # ieh22, jeh22 = np.where(e_h==4)
    # ieh_b, jeh_b = np.where(e_h==(-1))
    
    
    # pt.plot_init(mrksze=16,lnewdth=1)
    # ax.scatter(j1,i1, marker='s', color=c_KIT_green, label='1')
    # ax.scatter(j2,i2, marker='s', color=c_KIT_red, label='2')
    # pt.plot_init(mrksze=8,lnewdth=1)
    # ax.scatter(jev11-0.5, iev11, marker='^', color=c_KIT_green, label='11')
    # ax.scatter(jev12-0.5, iev12, marker='^', color=c_KIT_blue, label='12')
    # ax.scatter(jev22-0.5, iev22, marker='^', color=c_KIT_red, label='22')
    # ax.scatter(jev_b-0.5, iev_b, marker='^', color='k', label='edge')
    # ax.scatter(jeh11, ieh11-0.5, marker='^', color=c_KIT_green)
    # ax.scatter(jeh12, ieh12-0.5, marker='^', color=c_KIT_blue)
    # ax.scatter(jeh22, ieh22-0.5, marker='^', color=c_KIT_red)
    # ax.scatter(jeh_b, ieh_b-0.5, marker='^', color='k')
    
    
# def break_something_random(e_v, e_h):
#     c = np.random.choice([0,1,2,3])
#     if c == 0:
#         # Break from left
#         i_init = np.random.choice(np.arange(1,e_h.shape[0])) 
#         j_init = 0
#         e_h[i_init, j_init] = -1
#     if c == 1:
#         # Break from right
#         i_init = np.random.choice(np.arange(1,e_h.shape[0])) 
#         j_init = -1
#         e_h[i_init, j_init] = -1
#     if c == 2:
#         # Break from top
#         j_init = np.random.choice(np.arange(1,e_v.shape[0])) 
#         i_init = 0
#         e_v[i_init, j_init] = -1
#     if c == 3:
#         # Break from bottom
#         j_init = np.random.choice(np.arange(1,e_v.shape[0])) 
#         i_init = -1
#         e_v[i_init, j_init] = -1
    
#     print('Broke bond at index ', i_init, j_init)
#     return e_v, e_h

    
# # Generate 2D grids for pivots and edges
# def generate_grids_2D(A, X1, X2):
#     A0 = float_gcd(A*X1, A*X2)
#     # N: array with total number of squares [1, 2] 
#     N = np.array([int(A*X1/A0), int(A*X2/A0)])
#     DIM = int(np.ceil(np.sqrt(np.sum(N))))
    
#     # B: array with total number of bonds [11, 12, 22]
#     B = np.zeros(3) 
    
#     p = np.ones((DIM,DIM))*(-1)
#     e_v = np.ones((DIM,DIM+1))*(-1)
#     e_h = np.ones((DIM+1,DIM))*(-1)
    
#     # temporary counter
#     N1 = N[0]
#     N2 = N[1]
#     for i in range(DIM):
#         for j in range(DIM):
#             # Select 1 or 2 for p
#             if N1>0 and N2>0:
#                 p[i,j] = np.random.choice([1,2], p=[X1,X2])
#             elif N1==0 and N2>0:
#                 p[i,j] = 2 
#             elif N1>0 and N2==0:
#                 p[i,j] = 1
                
#             if i>0:
#                 e_h[i,j]=p[i,j]*p[i-1,j]
#             if j>0:
#                 e_v[i,j]=p[i,j]*p[i,j-1]
            
#             # Adjust counter
#             N1 -= int(p[i,j]==1)
#             N2 -= int(p[i,j]==2)
            
#             B += np.array([int(e_h[i,j]==1),int(e_h[i,j]==2),int(e_h[i,j]==4)]) 
#             B += np.array([int(e_v[i,j]==1),int(e_v[i,j]==2),int(e_v[i,j]==4)])                      
            
#     return p, e_v, e_h, N, B, A0