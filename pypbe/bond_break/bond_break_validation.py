# -*- coding: utf-8 -*-
"""
Validate 1d-MC-Bond-Breakage-Model
"""
import os,sys
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),"../.."))
from pypbe.dpbe import population as pop
from pypbe.bond_break.bond_break_jit import MC_breakage
from pypbe.utils.func.jit_pop import breakage_func_1d
from pypbe.bond_break.bond_break_generate_data import generate_complete_1d_data
from pypbe.bond_break.bond_break_post import 

import utils.plotter.plotter as pt   
from utils.plotter.KIT_cmap import c_KIT_green, c_KIT_red, c_KIT_blue, KIT_black_green_white
import matplotlib.pyplot as plt

def validate_frag_dist_1d(A, A0, STR, NO_FRAG, int_bre, N_GRIDS, N_FRACS, func_type,
                          v=2, q=1):
    X1 = 1
    X2 = 0
    STR = np.array([1,1,1])
    
    F = MC_breakage(A, X1, X2, STR, NO_FRAG, int_bre, N_GRIDS, N_FRACS, A0)
    
    pt.plot_init(mrksze=12,lnewdth=1)
    fig=plt.figure()    
    ax1=fig.add_subplot(1,2,1)
    ax2=fig.add_subplot(1,2,2)
    ax1, fig, _, _ = pt.plot_1d_hist(x=F[:,0],bins=100,scale='lin',xlbl='fragment size / a.u.',
                                         ylbl='Counts / $-$',clr=c_KIT_green,norm=False, alpha=0.7)
    y = A
    x_var = np.linspace(0, y, 40)
    q0 = breakage_func_1d(x_var, y, v, q, BREAKFVAL)
    ax2, fig = pt.plot_data(x_var, q0, fig=fig, ax=ax2,
                            lbl="q0",clr=c_KIT_green,mrk='^')
    
def validate_pbe(NS, S, A_rel, STR, NO_FRAG, int_bre, N_GRIDS, N_FRACS, output_dir, 
                 unc_type, v=2, q=1):
    p = calc_pbe(NS, S, dim=1)
    
    int_B_F = np.zeros((NS, NS))
    intx_B_F = np.zeros((NS, NS))
    V = np.copy(p.V)
    V_e = np.copy(p.V_e)
    V_e[0] = 0.0
    NO_TESTS = N_GRIDS*N_FRACS
    generate_complete_1d_data(NS,S,A_rel,STR,NO_FRAG,int_bre,N_GRIDS,N_FRACS,output_dir)
    for i in range(1, NS):
        file_name = f"{STR[0]}_{NO_FRAG}_{int_bre}_i{idx}.npy"      
        file_path = os.path.join(output_dir,file_name)
        F = np.load(file_path,allow_pickle=True)
        counts, x_vol_sum, _ =  calc_int_BF(NO_TESTS, F[:,0], V_e)
        int_B_F[:, i] = counts
        intx_B_F[:,i] = x_vol_sum
    output_file = "int_B_F_data"
    save_path = os.path.join(output_dir, "int_B_F.npz")
    np.savez(save_path,
             STR=STR,
             NO_FRAG=NO_FRAG,
             int_B_F=int_B_F,
             intx_B_F = intx_B_F)
    
    p = calc_pbe(NS, S, dim, True, save_path)
      
def calc_pbe(NS, S, dim, USE_MC_BOND=False,PTH_MC_BOND=None):
    p = pop(dim=dim)
    ## Set the PBE parameters
    t_vec = np.arange(0, 3601, 100, dtype=float)
    # Note that it must correspond to the settings of MC-Bond-Break.
    p.NS = NS
    p.S = S
    
    p.BREAKRVAL= 4
    p.BREAKFVAL= 5
    p.aggl_crit= 100
    p.process_type= "breakage"
    p.pl_v= 0.8
    p.pl_P1= 1e-3
    p.pl_P2= 2
    p.pl_P3= 1e-3
    p.pl_P4= 2
    p.COLEVAL= 2
    p.EFFEVAL= 1
    p.SIZEEVAL= 1
    if dim == 2:
        p.alpha_prim = np.array([1, 1, 1, 1])
    elif dim == 1:
        p.alpha_prim = 1
    p.CORR_BETA= 1e-4
    ## The original value is the particle size at 1% of the PSD distribution. 
    ## The position of this value in the coordinate system can be adjusted by multiplying by size_scale.
    size_scale = 1e-1
    p.R01 = 8.677468940430804e-07*size_scale
    p.R03 = 8.677468940430804e-07*size_scale
    
    ## If you need to read PSD data as initial conditions, set the PSD data path
    if p.process_type == 'breakage':
        p.USE_PSD = False
    else:
        p.USE_PSD = True
        p.DIST1 = os.path.join(p.pth,'data','PSD_data','PSD_x50_2.0E-6_RelSigmaV_1.5E-1.npy')
        p.DIST3 = os.path.join(p.pth,'data','PSD_data','PSD_x50_2.0E-6_RelSigmaV_1.5E-1.npy')
    
    ## Use the breakage function calculated by the MC-Bond-Break method
    p.USE_MC_BOND = USE_MC_BOND
    p.PTH_MC_BOND = PTH_MC_BOND
    p.solver = "ivp"
    
    ## Initialize the PBE
    p.V_unit = 1e-15
    p.full_init(calc_alpha=False)
    ## solve the PBE
    p.solve_PBE(t_vec=t_vec)
    N = p.N
    V_p = p.V
    
    if dim == 2:
        N0 = N[:,:,0]
        NE = N[:,:,-1]
    elif dim == 1:
       N0 = N[:,0]
       NE = N[:,-1]
    print('### Total Volume before and after..')
    print(np.sum(N0*V_p), np.sum(NE*V_p))
    return p
    
if __name__ == '__main__':
    NS = 10
    S = 2
    A = 100
    A0 = 1
    A_rel = 10
    X1 = 1
    X2 = 1 - X1
    STR = np.array([1,0.5,0.3])
    NO_FRAG = 4
    int_bre = 0
    NO_FRAG = 200
    N_GRIDS = 100
    ## BREAKFVAL = 1: Conservation of Hypervolume, random breakage into four fragments
    ## BREAKFVAL = 2: Conservation of First-Order Moments, random breakage into two fragments
    ## BREAKFVAL = 3: product function of power law.
    ## BREAKFVAL = 4: simple function of power law.
    ## BREAKFVAL = 5: Parabolic   
    BREAKFVAL = 5
    output_dir = os.path.join(os.path.dirname( __file__ ), "validation")
    
    
    
    
    
    