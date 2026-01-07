# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:44:42 2024

@author: xy0264
"""
import numpy as np
from copy import deepcopy
from numba import jit
from numba.typed import List as nb_List
import time
import matplotlib.pyplot as plt
import optframework.utils.plotter.plotter as pt
from optframework.utils.plotter.KIT_cmap import c_KIT_green, c_KIT_red, c_KIT_blue, KIT_black_green_white

# Allow closed-loop fragments to form inside particles, but there are still bugs!
@jit(nopython=True)
def MC_breakage_intern_frag(A, X1, X2, STR, NO_FRAG, int_bre=0, N_GRIDS=100, N_FRACS=100, A0=0, 
                init_break_random=False, gamma=1.0, aspect_ratio=1.0):
    """Perform a 2D, 2 material Monte Carlo breakage simulation.
    
    Parameters
    ----------
    A : float
        Total area of the agglomerate to break. (Current assumption: Use volume).
    X1 : float
        Area-fraction (volume-fraction) of material 1
    X2 : float
        Area-fraction (volume-fraction) of material 2
    STR : ndarray
        Bond strengths between materials in this order: [11, 12, 22]
    NO_FRAG : int
        Number of fragments per breakage event
    N_GRIDS : int, optional
        Number of simulated grids 
    N_FRACS : int, optional
        Number of simulated fractured (each has NO_FRAG fragments)        
    A0 : float, optional
        Area of smallest entity (smalles "primary particle")
        If A0==0: Calculate largest common divisor (based on certain tolerance)
    init_break_random : bool, optional
        True: First bond fracture is random (not material specific)
        False (default): Also use STR for selection
    
    
    Returns
    -------
    F : ndarray
        Fragment array. 
        Axis 0: All fragments
        Axis 1: [total area, X1, X2, fracture energy]
    
    """
    # Initialize fracture array (return)
    F = np.zeros((N_GRIDS*N_FRACS*NO_FRAG,4))
    
    if int_bre < 0 or int_bre > 1:
        raise Exception("int_bre is the relative depth relative to the particle size, ranging from [0,1]")
    
    # Loop through all grids based on A, X1 and X2
    for g in range(N_GRIDS):
        # print(f'Calculating grid no. {g+1}/{N_GRIDS}')
        # Generate grid and copy it (identical initial conditions for other fractures)
        # start_time_grid = time.time()
        
        G, N, B, A0, R, ibl = generate_grid_2D(A, X1, X2, int_bre, A0=A0)
        G0 = np.copy(G)
        
        for f in range(N_FRACS):
            # start_time_frac = time.time()
            # Tracking number of fragments and index history of all fragments
            no_frag = 1
            G = np.copy(G0)
            fracture_energy = 0
            G_old = np.copy(G)
            while no_frag < NO_FRAG:  
                # Initialize a new fracture. idx=None indicates that this is the first event
                G, idx, ff, str_bond, cur_dir, internal_start, first_crack, internal_idx, idx_new_hist = break_one_bond(G, STR, ibl, idx=None, init_break_random=init_break_random)
                # For each fracture keep a separate history (otherwise fragments cannot break "inside" themselves)
                idx_hist = nb_List([np.copy(internal_idx)])
                # Record the initial fracture points.
                idx_hist.extend(idx_new_hist)
                fracture_energy += str_bond
                
                if internal_start:
                    # If the first fracture already reaches the boundary
                    if not first_crack:
                        idx = np.copy(internal_idx)
                        ibl_tem = ibl
                    else:
                        ibl_tem = 1
                else:
                    ibl_tem = 1
                        
                # Pursue this fracture until it breaks through 
                # Rare cases lead to an endless loop (despite check_deadend call)
                # In this case simply repeat the fracture process from the beginning!
                cnt = 0
                while ff is False and cnt < G.shape[0]**2:
                    first_crack_tem = first_crack
                    G_tmp, idx_tmp, ff_tmp, str_bond, cur_dir, internal_start, first_crack, _, idx_new_hist = break_one_bond(G, STR, ibl_tem, idx=idx, gamma=gamma, 
                                                                                               prev_dir=cur_dir, internal_start=internal_start,
                                                                                               first_crack=first_crack)
                    
                    # Final Fracture is always valid
                    # Check for circular fracture (if not so, keep the result)
                    # Also check surrounding nodes for circular fracture (endless loop otherwise / deadend)
                    if check_idx_hist(idx_tmp, idx_hist): 
                        ff_tmp = True
                    idx = np.copy(idx_tmp)
                    G = np.copy(G_tmp)
                    ff = ff_tmp
                    idx_hist.extend(idx_new_hist)
                    fracture_energy += str_bond
                    ibl_tem = 1
                    if first_crack_tem and (not first_crack):
                        idx = np.copy(internal_idx)
                        ibl_tem = ibl
                    cnt += 1
                
                # Caught in an endless loop. Report and restart the fragmentation (reset no_frag)
                if cnt >= G.shape[0]**2: 
                    # no_frag = 1
                    G = np.copy(G_old)  
                else:
                    # Increase number of fragments and append to overall history
                    no_frag += 1
                if no_frag == NO_FRAG:
                    G_new = np.copy(G)
                    # Analyze framents 
                    G_new, cnt_1_tmp, cnt_2_tmp, val_arr = analyze_fragments(G_new)
                    if len(cnt_1_tmp) < NO_FRAG or len(cnt_2_tmp) < NO_FRAG:
                        no_frag -= 1
                        G_old = np.copy(G)
                    elif len(cnt_1_tmp) > NO_FRAG or len(cnt_2_tmp) > NO_FRAG:
                        no_frag -= 1
                        G = np.copy(G_old) 
                else:
                    G_old = np.copy(G)
            
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
            # elapsed_time_frac = time.time() - start_time_frac
            # print(f"A frac takes：{elapsed_time_frac} seconds")
        # elapsed_time_grid = time.time() - start_time_grid
        # print(f"A grid takes：{elapsed_time_grid} seconds")
                
    return F

@jit(nopython=True)
def MC_breakage(A, X1, X2, STR, NO_FRAG, int_bre=0, N_GRIDS=100, N_FRACS=100, A0=0, 
                init_break_random=False, gamma=1.0, aspect_ratio=1.0):
    """Perform a 2D, 2 material Monte Carlo breakage simulation.
    
    Parameters
    ----------
    A : float
        Total area of the agglomerate to break. (Current assumption: Use volume).
    X1 : float
        Area-fraction (volume-fraction) of material 1
    X2 : float
        Area-fraction (volume-fraction) of material 2
    STR : ndarray
        Bond strengths between materials in this order: [11, 12, 22]
    NO_FRAG : int
        Number of fragments per breakage event
    N_GRIDS : int, optional
        Number of simulated grids 
    N_FRACS : int, optional
        Number of simulated fractured (each has NO_FRAG fragments)        
    A0 : float, optional
        Area of smallest entity (smalles "primary particle")
        If A0==0: Calculate largest common divisor (based on certain tolerance)
    init_break_random : bool, optional
        True: First bond fracture is random (not material specific)
        False (default): Also use STR for selection
    
    
    Returns
    -------
    F : ndarray
        Fragment array. 
        Axis 0: All fragments
        Axis 1: [total area, X1, X2, fracture energy]
    
    """
    # Initialize fracture array (return)
    F = np.zeros((N_GRIDS*N_FRACS*NO_FRAG,4))
    
    if int_bre < 0 or int_bre > 1:
        raise Exception("int_bre is the relative depth relative to the particle size, ranging from [0,1]")
    
    # Loop through all grids based on A, X1 and X2
    for g in range(N_GRIDS):
        # print(f'Calculating grid no. {g+1}/{N_GRIDS}')
        # Generate grid and copy it (identical initial conditions for other fractures)
        # start_time_grid = time.time()
        
        G, N, B, A0, R, ibl = generate_grid_2D(A, X1, X2, int_bre, A0=A0)
        G0 = np.copy(G)
        
        for f in range(N_FRACS):
            # start_time_frac = time.time()
            # Tracking number of fragments and index history of all fragments
            no_frag = 1
            G = np.copy(G0)
            fracture_energy = 0
            G_old = np.copy(G)
            while no_frag < NO_FRAG:  
                # Initialize a new fracture. idx=None indicates that this is the first event
                G, idx, ff, str_bond, cur_dir, internal_start, first_crack, internal_idx, idx_new_hist = break_one_bond(G, STR, ibl, idx=None, init_break_random=init_break_random)
                # For each fracture keep a separate history (otherwise fragments cannot break "inside" themselves)
                idx_hist = nb_List([np.copy(internal_idx)])
                # Record the initial fracture points.
                idx_hist.extend(idx_new_hist)
                fracture_energy += str_bond
                
                if internal_start:
                    # If the first fracture already reaches the boundary
                    if not first_crack:
                        idx = np.copy(internal_idx)
                        ibl_tem = ibl
                    else:
                        ibl_tem = 1
                else:
                    ibl_tem = 1
                        
                # Pursue this fracture until it breaks through 
                # Rare cases lead to an endless loop (despite check_deadend call)
                # In this case simply repeat the fracture process from the beginning!
                cnt = 0
                while ff is False and cnt < G.shape[0]**2:
                    first_crack_tem = first_crack
                    G_tmp, idx_tmp, ff_tmp, str_bond, cur_dir, internal_start, first_crack, _, idx_new_hist = break_one_bond(G, STR, ibl_tem, idx=idx, gamma=gamma, 
                                                                                               prev_dir=cur_dir, internal_start=internal_start,
                                                                                               first_crack=first_crack)
                    
                    # Final Fracture is always valid
                    # Check for circular fracture (if not so, keep the result)
                    # Also check surrounding nodes for circular fracture (endless loop otherwise / deadend)
                    if not check_idx_hist(idx_tmp, idx_hist) \
                        and not check_deadend(idx_tmp, idx_hist, G): 
                        
                        idx = np.copy(idx_tmp)
                        G = np.copy(G_tmp)
                        ff = ff_tmp
                        idx_hist.extend(idx_new_hist)
                        fracture_energy += str_bond
                        ibl_tem = 1
                        if first_crack_tem and (not first_crack):
                            idx = np.copy(internal_idx)
                            ibl_tem = ibl
                    else:
                        # Prevent the first crack forms a circular loop
                        if first_crack_tem and (not first_crack):
                            first_crack = True
                    cnt += 1
                
                # Caught in an endless loop. Report and restart the fragmentation (reset no_frag)
                if cnt >= G.shape[0]**2: 
                    # no_frag = 1
                    G = np.copy(G_old)  
                else:
                    # Increase number of fragments and append to overall history
                    no_frag += 1
                    G_old = np.copy(G)
            
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
            # elapsed_time_frac = time.time() - start_time_frac
            # print(f"A frac takes：{elapsed_time_frac} seconds")
        # elapsed_time_grid = time.time() - start_time_grid
        # print(f"A grid takes：{elapsed_time_grid} seconds")
                
    return F

@jit(nopython=True)
def rand_choice_nb(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]

@jit(nopython=True)
def float_gcd(a, b, rtol = 1e-3, atol = 1e-8):
    t = min(abs(a), abs(b))
    while abs(b) > rtol * t + atol:
        a, b = b, a % b
    return a

# Generate 2D grid containing both pivots and edges
@jit(nopython=True)
def generate_grid_2D(A, X1, X2, int_bre, A0=0, aspect_ratio=1.0):
    """
    Generate a two-dimensional grid to represent the distribution 
    of two materials and the contact (bond) between them. The distribution 
    is randomly assigned based on the volume fraction X1 und X2 of the two materials.
    
    Parameters
    ----------
    A : float
        Total area of the agglomerate
    X1, X2 : float
        Volume fractions of material 1 and 2
    int_bre : float
        Internal breakage depth ratio
    A0 : float, optional
        Area of smallest unit (auto-calculated if 0)
    aspect_ratio : float, optional
        Width/Height ratio for rectangular particle shape (default=1.0 for square)
        aspect_ratio > 1.0: wider than tall
        aspect_ratio < 1.0: taller than wide
    """
    if A0 == 0 or (A0>(A*X1/2) and A0>(A*X2/2)):
        A0 = float_gcd(A*X1, A*X2)

    # N: array with total number of squares [1, 2]
    # R: array with rest area [1, 2]
    N = np.array([int((A*X1)//A0), int((A*X2)//A0)]) 
    R = np.array([(A*X1)%A0, (A*X2)%A0])
    
    # Calculate dimensions for rectangular grid
    total_units = np.sum(N)
    if aspect_ratio >= 1.0:
        # Wider than tall
        DIM_WIDTH = int(np.ceil(np.sqrt(total_units * aspect_ratio)))
        DIM_HEIGHT = int(np.ceil(total_units / DIM_WIDTH))
    else:
        # Taller than wide
        DIM_HEIGHT = int(np.ceil(np.sqrt(total_units / aspect_ratio)))
        DIM_WIDTH = int(np.ceil(total_units / DIM_HEIGHT))
    
    # Ensure we have enough space for all units
    while DIM_WIDTH * DIM_HEIGHT < total_units:
        if aspect_ratio >= 1.0:
            DIM_WIDTH += 1
        else:
            DIM_HEIGHT += 1
    
    # B: array with total number of bonds [11, 12, 22]
    B = np.zeros(3) 
    
    # G: Grid containing both pivots (squares), bonds and edges
    # pivots: 1 or 2 (material 1 or 2)
    # bonds: -1, 11, 12 or 22 (combinations of 1 and 2)
    #           -1 is an "outside" surface with no contact
    # edges: 0
    G = np.ones((2*DIM_HEIGHT+1, 2*DIM_WIDTH+1))*(-1)
    if int_bre == 0:
        int_bre_len = 1
    else:
        int_bre_len = int(np.ceil(max(DIM_HEIGHT, DIM_WIDTH) * int_bre))
    
    # Set edges (starting at [0,0] in steps of 2)
    for i in range(0, 2*DIM_HEIGHT+1, 2):
        for j in range(0, 2*DIM_WIDTH+1, 2):
            G[i,j] = 0
    
    # temporary counter
    N1 = N[0]
    N2 = N[1]
    # Loop over all PIVOTS. They start at [1,1] and go in index steps of 2 (bond in between)
    for i in range(1, 2*DIM_HEIGHT, 2):
        for j in range(1, 2*DIM_WIDTH, 2):
            # Only place material if we haven't exceeded our unit count
            if N1 + N2 <= 0:
                break
                
            # Set pivot
            if N1>0 and N2>0:
                # G[i,j] = np.random.choice([1,2], p=[X1,X2])                
                G[i,j] = rand_choice_nb(np.array([1,2]), np.array([X1,X2]))
            elif N1==0 and N2>0:
                G[i,j] = 2 
            elif N1>0 and N2==0:
                G[i,j] = 1
            else:
                # No more materials to place
                break
                        
            # Adjust counter
            N1 -= int(G[i,j]==1)
            N2 -= int(G[i,j]==2)
            
            # Set bonds and update counter (only if both materials exist)
            if i>1 and G[i-2,j] in [1, 2] and G[i,j] in [1, 2]:
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

            if j>1 and G[i,j-2] in [1, 2] and G[i,j] in [1, 2]:
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
            
    return G, N, B, A0, R, int_bre_len

@jit(nopython=True)
def break_one_bond(G, STR, ibl=1, idx=None, init_break_random=False, gamma=1.0, 
                   prev_dir=-1, internal_start=False, first_crack=False):
    """
    Simulates the process of breaking a bond in a given 2D mesh.
    
    Parameters
    ----------
    G : 2d-array
        Represents the mesh model of the material, which contains the material's pivot unit (1 or 2), 
        connected bonds (-1, 11, 12, 22), and edges (0).
    STR : 1d-array
        Contains the strength of different types of bonds (11, 12, 22). 
        These values ​​are used to calculate the probability of breakage.
    idx : np.array([i,j])
        Indicating the index of current edge to propagate breakge.
        None indicates that we start a new rupture (from the outside edge)
        
    Returns
    -------
    G_new : 2d-array
        The updated mesh, showing the state after fracture.
    idx_new : np.array([i,j])
        The index of the new fracture edge, providing a starting point for the next fracture.
    fracture_flag : Bool
        Indicating whether a complete fracture has occurred.
    str_array[b_idx] : float
        The strength of the broken connection.
    """
    
    G_new = np.copy(G)
    nrows, ncols = G.shape
    internal_idx = np.array([-1,-1])
    # Find initial edge to start rupture if idx=None
    # A valid edge must only have one or two neighboring "real" bonds (not -1)
    if idx is None:
        # 允许从网格内任何一个值为0且至少有一个可断裂键的单元格开始
        valid_points = []
        for i in range(0, nrows, 2):
            for j in range(0, ncols, 2):
                if G[i, j] == 0:
                    # Count how many real bonds around
                    cnt_bonds = 0
                    if i > 0   and G[i-1, j] != -1: cnt_bonds += 1
                    if i <nrows-1 and G[i+1, j] != -1: cnt_bonds += 1
                    if j > 0   and G[i, j-1] != -1: cnt_bonds += 1
                    if j <ncols-1 and G[i, j+1] != -1: cnt_bonds += 1
                    # 只要周围存在>=1个可断裂键，就可以作为起裂点
                    if cnt_bonds >= 1:
                        valid_points.append((i,j, cnt_bonds))
        # 随机选一个点
        idx_choice = np.random.randint(0, len(valid_points))
        idx0, idx1, bond_count = valid_points[idx_choice]
        idx = np.array([idx0, idx1])
        # 若该点周围键>2，视为“内部”
        if bond_count > 2:
            internal_start = True
            first_crack = True
        internal_idx = idx
        

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
    # To describe the redistribution of stress due to crack expansion, 
    # it is assumed that there is stress concentration in the direction of 
    # the crack advancement.
    if prev_dir != -1 and p[prev_dir] > 0:
        p[prev_dir] *= gamma
        
    # Normalize probabilites
    p /= np.sum(p)
    
    # Select a bond to break
    # b_idx = np.random.choice(np.arange(4), p=p)  
    b_idx = rand_choice_nb(np.arange(4), p)
    # Save the direction of the breaking, and use it as prev_dir next time
    cur_dir = b_idx
    
    # for l in range(ibl):
    #     if b_idx == 0:
    #         G_new[idx[0]-(1+2*l),idx[1]] = -1
    #         idx_new = np.array([idx[0]-(2+2*l),idx[1]]) 
    #         if l == 0:
    #             idx_new_hist = nb_List([np.copy(idx_new)])
    #         else:
    #             idx_new_hist.append(np.copy(idx_new))
    #         if (G_new[idx_new[0], idx_new[1]+1] == -1 and G_new[idx_new[0], idx_new[1]-1] == -1) or G_new[idx_new[0]-1, idx_new[1]] == -1:
    #             break
    #         # idx[0] -= 2
    #     if b_idx == 1:
    #         G_new[idx[0]+1+2*l,idx[1]] = -1
    #         idx_new = np.array([idx[0]+2+2*l,idx[1]])
    #         if l == 0:
    #             idx_new_hist = nb_List([np.copy(idx_new)])
    #         else:
    #             idx_new_hist.append(np.copy(idx_new))
    #         if (G_new[idx_new[0], idx_new[1]+1] == -1 and G_new[idx_new[0], idx_new[1]-1] == -1) or G_new[idx_new[0]+1, idx_new[1]] == -1:
    #             break
    #         # idx[0] += 2
    #     if b_idx == 2:
    #         G_new[idx[0],idx[1]-(1+2*l)] = -1
    #         idx_new = np.array([idx[0],idx[1]-(2+2*l)])
    #         if l == 0:
    #             idx_new_hist = nb_List([np.copy(idx_new)])
    #         else:
    #             idx_new_hist.append(np.copy(idx_new))
    #         if (G_new[idx_new[0]+1, idx_new[1]] == -1 and G_new[idx_new[0]-1, idx_new[1]] == -1) or G_new[idx_new[0], idx_new[1]-1] == -1:
    #             break
    #         # idx[1] -= 2
    #     if b_idx == 3:
    #         G_new[idx[0],idx[1]+1+2*l] = -1
    #         idx_new = np.array([idx[0],idx[1]+2+2*l])
    #         if l == 0:
    #             idx_new_hist = nb_List([np.copy(idx_new)])
    #         else:
    #             idx_new_hist.append(np.copy(idx_new))
    #         if (G_new[idx_new[0]+1, idx_new[1]] == -1 and G_new[idx_new[0]-1, idx_new[1]] == -1) or G_new[idx_new[0], idx_new[1]+1] == -1:
    #             break
            # idx[1] += 2

    # Progress the fracture
    direction_map = [(-1, 0),  # left
                     (1, 0),   # right
                     (0, -1),  # down
                     (0, 1)    # up
                     ]
    # idx_new_hist = nb_List([np.array((-1,-1))])
    for l in range(ibl):
        dx, dy = direction_map[b_idx]  
        G_new[idx[0] + (dx * (1 + 2 * l)), idx[1] + (dy * (1 + 2 * l))] = -1
        
        # update idx_new
        idx_new = np.array([idx[0] + (dx * (2 + 2 * l)), idx[1] + (dy * (2 + 2 * l))])
        
        # recorde the fracture point
        if l == 0:
            idx_new_hist = nb_List([np.copy(idx_new)])
        else:
            idx_new_hist.append(np.copy(idx_new))
    
        # Check whether the loop is to be terminated (whether the boundary is encountered)
        if ((G_new[idx_new[0] + (dx == 0), idx_new[1] + (dy == 0)] == -1 and 
             G_new[idx_new[0] - (dx == 0), idx_new[1] - (dy == 0)] == -1) or 
            G_new[idx_new[0] + dx, idx_new[1] + dy] == -1):
            break
        
    # Check if this leads to a complete fracture (new index has not more than 1 breakable bonds)
    # Counter for valid bonds
    cnt = 0
    fracture_flag = False
    
    # Check surrounding bonds
    # Check whether it has reached the border or existing cracks
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
    
    if cnt <= 1:
        if internal_start and first_crack:
            first_crack = False
        else:
            fracture_flag = True
    # print(f"{first_crack} and {fracture_flag}")
    # if cnt>1:
    #     fracture_flag = False
    # else:
    #     fracture_flag = True
        
    return G_new, idx_new, fracture_flag, str_array[b_idx], cur_dir, internal_start, first_crack, internal_idx, idx_new_hist

@jit(nopython=True)
def calculate_probabilities(G, init_array, STR):
    lenth = int(init_array.shape[1]/2)
    for cnt in range(init_array.shape[1]):
        for l in range(init_array.shape[0]):
            ## Check if the break has reached the boundary of the current fragment
            if init_array[l, cnt, 1] == -1:
                init_array[l:, cnt, 4] = 0
                break
            if l > 0:
                if cnt < lenth: # Bottom and top row keys
                    i, j = int(init_array[l, cnt, 2]), int(init_array[l, cnt, 3])
                    if G[i-1, j] == -1 and G[i+1, j] == -1:
                        init_array[l:, cnt, 4] = 0
                        break
                else:  # Left and right column keys
                    i, j = int(init_array[l, cnt, 2]), int(init_array[l, cnt, 3])
                    if G[i, j-1] == -1 and G[i, j+1] == -1:
                        init_array[l:, cnt, 4] = 0
                        break
            ## Since the calculation should be based on the sum of the strengths, 
            ## their strengths are saved here first.
            ## Afterwards, the inverse of the sum of the intensities is calculated 
            ## as the fracture probability. 
            if init_array[l, cnt, 1] == 11:
                init_array[l, cnt, 4] = STR[0]
            elif init_array[l, cnt, 1] == 12:
                init_array[l, cnt, 4] = STR[1]
            elif init_array[l, cnt, 1] == 22:
                init_array[l, cnt, 4] = STR[2]
    ## Rejecting the starting position where the fracture should not occur            
    zero_prob_indices = []
    for cnt in range(init_array.shape[1]):
        if init_array[0, cnt, 4] == 0:
            zero_prob_indices.append(cnt)

    for l in range(init_array.shape[0]):
        for idx in zero_prob_indices:
            init_array[l, idx, 4] = 0
    
    ## Calculate the sum of the strengths of all bonds along possible breaking directions
    str_sum = init_array[:,:,4].sum(axis=0)
    init_array_prob = np.zeros_like(str_sum)
    init_array_prob[np.where(str_sum != 0)] = 1 / str_sum[np.where(str_sum != 0)] 
    init_array_prob /= np.sum(init_array_prob)
    
    return str_sum, init_array_prob

@jit(nopython=True)
def recursive_fun(G, i, j, cnt_1, cnt_2, new_value=0):
    nrows, ncols = G.shape
    
    if i<0 or i>=nrows or j<0 or j>=ncols or (G[i,j]!=1 and G[i,j]!=2):
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

@jit(nopython=True)
def iterative_fun(G, i, j, new_value=0):
    nrows, ncols = G.shape
    stack = [(i, j)]
    cnt_1 = cnt_2 = 0
    
    while stack:
        ci, cj = stack.pop()
        if ci < 0 or ci >= nrows or cj < 0 or cj >= ncols or G[ci, cj] not in (1, 2):
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
            
            if 0 <= ni < nrows and 0 <= nj < ncols:
                if G[conn_i, conn_j] not in (-1, -2):  # Check if connection is not broken
                    stack.append((ni, nj))
                    G[conn_i, conn_j] = -2  # Mark the connection as checked
    
    return G, cnt_1, cnt_2

@jit(nopython=True)      
def analyze_fragments(G):
    nrows, ncols = G.shape
    
    cnt_1_arr = []
    cnt_2_arr = []
    val_arr = []
    val_cnt = 0
    for i in range(1,nrows,2):
        for j in range(1,ncols,2):
            if G[i,j]==1 or G[i,j]==2:
                # G, cnt_1, cnt_2 = recursive_fun(G, i, j, 0, 0, new_value=3+val_cnt) 
                G, cnt_1, cnt_2 = iterative_fun(G, i, j, new_value=3+val_cnt) 
                cnt_1_arr.append(cnt_1)
                cnt_2_arr.append(cnt_2)
                val_arr.append(3+val_cnt)
                val_cnt += 1
                
    return G, np.array(cnt_1_arr), np.array(cnt_2_arr), val_arr 
              
@jit(nopython=True)
def check_idx_hist(idx, idx_hist):
    # idx is numpy array
    # idx_hist is numpy array of numpy arrays
    # return True if idx is found inside idx_hist
    for i in range(len(idx_hist)):
        if np.all(idx == idx_hist[i]):
            return True    
    return False

@jit(nopython=True)
def check_deadend(idx, idx_hist, G):
    # A deadend is reached (return True) if all surrounding pivots are in idx_hist       
    bool_list = np.array([check_idx_hist(idx+np.array([2,0]), idx_hist), 
                          check_idx_hist(idx-np.array([2,0]), idx_hist),
                          check_idx_hist(idx+np.array([0,2]), idx_hist),
                          check_idx_hist(idx-np.array([0,2]), idx_hist)])
    
    return np.all(bool_list)

# # Allow closed-loop fragments to form inside particles, but there are still bugs!
def single_sim_intern_frag(A, X1, X2, STR, NO_FRAG, int_bre, A0=None, init_break_random=False, plot=True, 
               close=False, verbose=True,gamma=1.0,aspect_ratio=1.0):

    if close: plt.close('all')
    
    # Generate and plot grid
    start_time = time.time()
    print("start generating grid")
    G, N, B, A0, R, ibl = generate_grid_2D(A, X1, X2, int_bre, A0=A0,aspect_ratio=aspect_ratio)
    G0 = np.copy(G)
    elapsed_time = time.time() - start_time
    start_time = time.time()
    print(f"The generation of grids takes：{elapsed_time} seconds")
    # print(A0, A*X1/A0, A*X2/A0)
    if plot: ax0, fig0 = plot_G(G0, title='Initial grid')
    
    # Breaking stuff
    # Tracking number of fragments and index history of all fragments
    no_frag = 1
    fracture_energy = 0
    G_old = np.copy(G)
    while no_frag < NO_FRAG:
        # if verbose: print(f'Starting fracture. Currently at {no_frag} fragments')
        # Initialize a new fracture. idx=None indicates that this is the first event
        G, idx, ff, str_bond, cur_dir, internal_start, first_crack, internal_idx, idx_new_hist = break_one_bond(
            G, STR, ibl, idx=None, init_break_random=init_break_random)
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print(f"The first breaking idx: {idx} , internal_start: {internal_start}, internal_idx: {internal_idx}")
        
        # For each fracture keep a separate history (otherwise fragments cannot break "inside" themselves)
        idx_hist = nb_List([np.copy(internal_idx)])
        # Record the initial fracture points.
        idx_hist.extend(idx_new_hist)
        fracture_energy += str_bond
        
        if internal_start:
            # If the first fracture reaches the boundary
            if not first_crack:
                idx = np.copy(internal_idx)
                ibl_tem = ibl
            else:
                ibl_tem = 1
        else:
            ibl_tem = 1
            
        # Pursue this fracture until it breaks through 
        # Rare cases lead to an endless loop (despite check_deadend call)
        # In this case simply repeat the fracture process from the beginning!
        cnt = 0
        while ff is False and cnt < 2*G.shape[0]:
            first_crack_tem = first_crack
            G_tmp, idx_tmp, ff_tmp, str_bond, cur_dir, internal_start, first_crack, _, idx_new_hist = break_one_bond(G, STR, 
                                                                                         ibl_tem, idx=idx, gamma=gamma, 
                                                                                       prev_dir=cur_dir, internal_start=internal_start,
                                                                                       first_crack=first_crack)
            print(f"internal start:{internal_start}, first_crack: {first_crack}, idx: {idx_tmp}")
            # Final Fracture is always valid
            # Check for circular fracture (if not so, keep the result)
            # Also check surrounding nodes for circular fracture (endless loop otherwise / deadend)
            if check_idx_hist(idx_tmp, idx_hist): 
                ff_tmp = True
            idx = np.copy(idx_tmp)
            G = np.copy(G_tmp)
            ff = ff_tmp
            idx_hist.extend(idx_new_hist)
            fracture_energy += str_bond
            ibl_tem = 1
            if first_crack_tem and (not first_crack):
                idx = np.copy(internal_idx)
                ibl_tem = ibl
            cnt += 1
        
        # Caught in an endless loop. Report and restart the fragmentation (reset no_frag and idx_hist_frag)
        if cnt >= 2*G.shape[0]: 
            G = np.copy(G_old)            
            if verbose: print('Caught in an endless loop :( Restarting this fragmentation process..')
        else:
            # Increase number of fragments and append to overall history
            no_frag += 1
            elapsed_time = time.time() - start_time
            start_time = time.time()
            # print(f"A fracture process takes：{elapsed_time} seconds")
            # Plot current fracture
            if plot: _, _ = plot_G(G, title=f'Currently {no_frag} fragments')  
    
    # Analyze framents (use copy to retain original G)
        if no_frag == NO_FRAG:
            G_new = np.copy(G)
            G_new, cnt_1_arr, cnt_2_arr, val_arr = analyze_fragments(G_new)
            if len(cnt_1_arr) < NO_FRAG or len(cnt_2_arr) < NO_FRAG:
                no_frag -= 1
                G_old = np.copy(G)
                print('There is a break event where no fragments are generated')
            elif len(cnt_1_arr) > NO_FRAG or len(cnt_2_arr) > NO_FRAG:
                no_frag -= 1
                G = np.copy(G_old)   
                print(f'Error: There are more than NO_FRAG being generated! {len(cnt_1_arr)}')
        else:
            G_old = np.copy(G)
    elapsed_time = time.time() - start_time
    # print(f"The analyze of framents takes：{elapsed_time} seconds")
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

def single_sim(A, X1, X2, STR, NO_FRAG, int_bre, A0=None, init_break_random=False, plot=True, 
               close=False, verbose=True,gamma=1.0,aspect_ratio=1.0):

    if close: plt.close('all')
    
    # Generate and plot grid
    start_time = time.time()
    print("start generating grid")
    G, N, B, A0, R, ibl = generate_grid_2D(A, X1, X2, int_bre, A0=A0, aspect_ratio=aspect_ratio)
    # M, Hb, Vb, meta, B = generate_compact_grid(A, X1, X2, A0=A0, aspect_ratio=aspect_ratio)
    # ibl = 1
    # G = to_old_G_layout(M, Hb, Vb)
    G0 = np.copy(G)
    elapsed_time = time.time() - start_time
    start_time = time.time()
    print(f"The generation of grids takes：{elapsed_time} seconds")
    # print(A0, A*X1/A0, A*X2/A0)
    if plot: ax0, fig0 = plot_G(G0, title='Initial grid')
    
    # Breaking stuff
    # Tracking number of fragments and index history of all fragments
    no_frag = 1
    fracture_energy = 0
    G_old = np.copy(G)
    while no_frag < NO_FRAG:
        # if verbose: print(f'Starting fracture. Currently at {no_frag} fragments')
        # Initialize a new fracture. idx=None indicates that this is the first event
        G, idx, ff, str_bond, cur_dir, internal_start, first_crack, internal_idx, idx_new_hist = break_one_bond(
            G, STR, ibl, idx=None, init_break_random=init_break_random)
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print(f"The first breaking idx: {idx} , internal_start: {internal_start}, internal_idx: {internal_idx}")
        
        # For each fracture keep a separate history (otherwise fragments cannot break "inside" themselves)
        idx_hist = nb_List([np.copy(internal_idx)])
        # Record the initial fracture points.
        idx_hist.extend(idx_new_hist)
        fracture_energy += str_bond
        
        if internal_start:
            # If the first fracture reaches the boundary
            if not first_crack:
                idx = np.copy(internal_idx)
                ibl_tem = ibl
            else:
                ibl_tem = 1
        else:
            ibl_tem = 1
            
        # Pursue this fracture until it breaks through 
        # Rare cases lead to an endless loop (despite check_deadend call)
        # In this case simply repeat the fracture process from the beginning!
        cnt = 0
        while ff is False and cnt < 2*G.shape[0]:
            first_crack_tem = first_crack
            G_tmp, idx_tmp, ff_tmp, str_bond, cur_dir, internal_start, first_crack, _, idx_new_hist = break_one_bond(G, STR, 
                                                                                         ibl_tem, idx=idx, gamma=gamma, 
                                                                                       prev_dir=cur_dir, internal_start=internal_start,
                                                                                       first_crack=first_crack)
            print(f"internal start:{internal_start}, first_crack: {first_crack}, idx: {idx_tmp}")
            # Final Fracture is always valid
            # Check for circular fracture (if not so, keep the result)
            # Also check surrounding nodes for circular fracture (endless loop otherwise / deadend)
            if not check_idx_hist(idx_tmp, idx_hist) \
                and not check_deadend(idx_tmp, idx_hist, G): 
                
                idx = np.copy(idx_tmp)
                G = np.copy(G_tmp)
                ff = ff_tmp
                idx_hist.extend(idx_new_hist)
                fracture_energy += str_bond
                ibl_tem = 1
                if first_crack_tem and (not first_crack):
                    idx = np.copy(internal_idx)
                    ibl_tem = ibl
                # print(f'valid event')
            else:
                # Prevent the first crack forms a circular loop
                if first_crack_tem and (not first_crack):
                    first_crack = True
                if verbose: print(f'index {idx_tmp} already inside idx_hist or in deadend')
            cnt += 1
        
        # Caught in an endless loop. Report and restart the fragmentation (reset no_frag and idx_hist_frag)
        if cnt >= 2*G.shape[0]: 
            G = np.copy(G_old)            
            if verbose: print('Caught in an endless loop :( Restarting this fragmentation process..')
        else:
            # Increase number of fragments and append to overall history
            no_frag += 1
            G_old = np.copy(G)
            elapsed_time = time.time() - start_time
            start_time = time.time()
            # print(f"A fracture process takes：{elapsed_time} seconds")
            # Plot current fracture
            if plot: _, _ = plot_G(G, title=f'Currently {no_frag} fragments')  
            
    G_new = np.copy(G)        
    G_new, cnt_1_arr, cnt_2_arr, val_arr = analyze_fragments(G_new)
    elapsed_time = time.time() - start_time
    # print(f"The analyze of framents takes：{elapsed_time} seconds")
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

# Plot any given G array (for debugging/visualization only)
def plot_G(G, title=None, fill_no=[], fill_clr=[]):    
    i_p1, j_p1 = np.where(G==1)
    i_p2, j_p2 = np.where(G==2)
    
    i_b11, j_b11 = np.where(G==11)
    i_b12, j_b12 = np.where(G==12)  
    i_b22, j_b22 = np.where(G==22) 
    i_bm1, j_bm1 = np.where(G==-1)
    
    i_e, j_e = np.where(G==0)
    
    fig=plt.figure(figsize=[5,5])    
    try:
        ax=fig.add_subplot(1,1,1) 
    except Exception as e:
        print("🔥 Error:", e)
        traceback.print_exc()
    
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

# Visualize the MC results
def plot_F(F):
    # 2D Fragment distribution
    ax1, fig1, cb1, H1, xe1, ye1 = pt.plot_2d_hist(x=F[:,0]*F[:,1],y=F[:,0]*F[:,2],bins=(20,20),w=None,
                                                   scale=('lin','lin'), clr=KIT_black_green_white.reversed(), 
                                                   xlbl='Partial Volume 1 $V_1$ / $\mathrm{m^3}$', norm=False,
                                                   ylbl='Partial Volume 2 $V_2$ / $\mathrm{m^3}$', grd=True,
                                                   scale_hist='log', hist_thr=1e-4)
    
    
    # 1D Histogram of fracture energy   
    ax2, fig2, H2, xe2 = pt.plot_1d_hist(x=F[:,3],bins=100,scale='lin',xlbl='Fracture Energy / a.u.',
                                         ylbl='Counts / $-$',clr=c_KIT_green,norm=False, alpha=0.7)
    #ax2.set_yscale('log')
    
    # 2D Histogram of fracture energy vs. fragment size
    ax3, fig3, cb3, H3, xe3, ye3 = pt.plot_2d_hist(x=F[:,0],y=F[:,3],bins=(20,20),w=None,
                                                   scale=('lin','lin'), clr=KIT_black_green_white.reversed(), 
                                                   xlbl='Fragment Size $V$ / $\mathrm{m^3}$', norm=False,
                                                   ylbl='Fracture Energy / a.u.', grd=True,
                                                   scale_hist='log', hist_thr=1e-4)
    
    # 1D Histogram of fracture energy   
    ax4, fig4, H4, xe4 = pt.plot_1d_hist(x=F[:,0],bins=100,scale='lin',xlbl='Fragment Size $V$ / $\mathrm{m^3}$',
                                         ylbl='Counts / $-$',clr=c_KIT_green,norm=False, alpha=0.7)
    return ax1, ax2, ax3, ax4

# %% MAIN    
if __name__ == '__main__':
    ########### -----------
    # import cProfile
    # import pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    ########### -----------
    
    A0 = 1.0
    A = 1000*A0
    X1 = 0.6
    X2 = 1-X1
    STR = np.array([1,1,1])
    NO_FRAG = 4
    int_bre = 0
    aspect_ratio = 2
    
    INIT_BREAK_RANDOM = False
    N_GRIDS, N_FRACS = 200, 200
    ## relative depth relative to the grid size, ranging from [0,1]
    
    # Perform stochastic simulation
    # Fragment array [total area, X1, X2, fracture energy]
    start_time = time.time()
    F = MC_breakage(A, X1, X2, STR, NO_FRAG, int_bre, N_GRIDS=N_GRIDS, N_FRACS=N_FRACS, 
                    A0=A0, init_break_random=INIT_BREAK_RANDOM) 
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The MC-Simulation takes：{elapsed_time:.2f} seconds")
    ########### -----------
    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.sort_stats('cumulative')
    # stats.print_stats(20)
    # print('#### Profiler of MC_breakage')
    # stats.print_stats('MC_breakage')
    ########### -----------
    ax1, ax2, ax3, ax4 = plot_F(F)
    # %% TESTS DEBUG
    TEST = False
    if TEST:
        import sys, os
        from matplotlib.colors import to_rgba
        import traceback
        pt.close()
        pt.plot_init(mrksze=12,lnewdth=1)
            
        # Visualize distributions
        # ax1, ax2, ax3, ax4 = plot_F(F)
        
        # Perform a single simulation (1 grid, 1 fracture) for visualization
        G, G_new, R, cnt_1_arr, cnt_2_arr, val_arr, fracture_energy, F_test = \
            single_sim(A, X1, X2, STR, NO_FRAG, int_bre, plot=True, A0=A0,
                        close=True, init_break_random=INIT_BREAK_RANDOM, aspect_ratio=aspect_ratio)
