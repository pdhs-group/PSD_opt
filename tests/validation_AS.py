# -*- coding: utf-8 -*-
"""
Validate discrete PBE with analytical solutions for various cases

@author: xy0264
"""
# %% IMPORTS
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import sys, os
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),".."))
import numpy as np
import math
import pypbe.utils.plotter.plotter as pt
from pypbe.utils.plotter.KIT_cmap import c_KIT_green, c_KIT_red, c_KIT_blue

from pypbe.dpbe import population as pop_disc
from pypbe.mcpbe import population_MC as pop_mc 

#%% CASES
def calculate_case(CASE, PBE=True, MC=False):
    # Initialize mu(i,j,t) matrix (3D, [i,j,t])
    mu_as = np.zeros((3,3,len(t)))
    mu_pbe = np.zeros((3,3,len(t)))
    mu_mc = np.zeros((3,3,len(t)))  
    std_mu_mc = np.zeros((3,3,len(t)))
    mu_tmp = []
    m_save = []
    p = None
    m = None
    
    #%%% '1D_const_mono': 1D, constant kernel, monodisperse initial conditions
    if CASE == '1D_const_mono':
        ### POPULATION BALANCE
        if PBE:
            p = pop_disc(1, disc=grid)
            
            p.process_type = process_type
            p.BREAKFVAL = BREAKFVAL
            p.BREAKRVAL = BREAKRVAL
            p.pl_v = pl_v   ## number of fragments
            p.pl_q = pl_q   ## parameter describes the breakage type
            p.pl_P1 = pl_P1
            p.pl_P2 = pl_P2
            p.G = G
            
            p.NS = NS  
            p.S = S
            p.COLEVAL = 3                          # Constant beta
            p.EFFEVAL = 2                           # Case for calculation of alpha
            p.CORR_BETA = beta0
            p.SIZEEVAL = 1
            p.R01 = x/2
            p.USE_PSD = False                  
            p.P1=0                                  # No breakage     
            p.N01 = n0
            
              
            p.calc_R()
            p.init_N()
            p.alpha_prim = 1
            p.calc_F_M()
            p.calc_B_R()
            p.calc_int_B_F()
        
            p.solve_PBE(t_vec = t)
            mu_pbe = p.calc_mom_t()
        
        if MC:
            # Calculate multiple times (stochastic process)
            mu_tmp = []
            for l in range(N_MC):
                m = pop_mc(1)
                m.c[0] = n0*v0
                m.x[0] = x
                m.PGV[0] = 'mono'
                m.BETACALC = 1
                m.beta0 = beta0
                m.a0 = a0
                m.VERBOSE = VERBOSE
                m.savesteps = len(t)
                m.tA = t[-1]
                
                m.init_calc()
                
                m.solve_MC()
                mu_tmp.append(m.calc_mom_t())
                m_save.append(m)
            
            # Mean and STD of moments for all N_MC loops
            mu_mc = np.mean(mu_tmp,axis=0)
            if N_MC > 1: std_mu_mc = np.std(mu_tmp,ddof=1,axis=0)
        
        ### ANALYTICAL SOLUTION FROM KUMAR DISSERTATION A.7
        if process_type == "agglomeration":
            mu_as[0,0,:] = 2*n0/(2+beta0*n0*t)
            mu_as[1,0,:] = np.ones(t.shape)*c 
        elif process_type == "breakage":
            # see Kumar Dissertation A.1
            N_as = np.zeros((NS,len(t)))
            V_sum = np.zeros((NS,len(t)))
            for i in range(NS):
                for j in range(len(t)):
                    if i != NS-1:
                        N_as[i,j] = (-(t[j]*p.V[-1]+1)+t[j]*p.V_e[i+1])*np.exp(-p.V_e[i+1]*t[j])-\
                            (-(t[j]*p.V[-1]+1)+t[j]*p.V_e[i])*np.exp(-p.V_e[i]*t[j])
                    else:
                        N_as[i,j] = (-(t[j]*p.V[-1]+1)+t[j]*p.V[i])*np.exp(-p.V[i]*t[j])-\
                            (-(t[j]*p.V[-1]+1)+t[j]*p.V_e[i])*np.exp(-p.V_e[i]*t[j]) + \
                            (np.exp(-t[j]*p.V[i]))
                    V_sum[i,j] = N_as[i,j] * p.V[i]
            mu_as[0,0,:] = N_as.sum(axis=0)
            mu_as[1,0,:] = np.ones(t.shape)*c 
        else:
            mu_as[0,0,:] = np.ones(t.shape)*c 
            mu_as[1,0,:] = np.ones(t.shape)*c 
    
    #%%% '2D_const_mono': 2D, constant kernel, monodisperse initial conditions
    elif CASE == '2D_const_mono':
        ### POPULATION BALANCE
        if PBE:
            p = pop_disc(2, disc=grid)
            
            p.process_type = process_type
            p.BREAKFVAL = BREAKFVAL
            p.BREAKRVAL = BREAKRVAL
            p.pl_v = pl_v   ## number of fragments
            p.pl_q = pl_q   ## parameter describes the breakage type
            p.pl_P1 = pl_P1
            p.pl_P2 = pl_P2
            p.G = G
            
            p.NS = NS  
            p.S = S
            p.COLEVAL = 3                           # Constant beta
            p.EFFEVAL = 2                           # Case for calculation of alpha
            p.CORR_BETA = beta0
            p.SIZEEVAL = 1
            p.R01, p.R03 = x/2, x/2
            p.USE_PSD = False                  
            p.P1=0                                  # No breakage     
            p.N01, p.N03 = n0, n0
    
            p.calc_R()
            p.init_N()
            p.alpha_prim = np.ones(4)
            p.calc_F_M()
            p.calc_B_R()
            p.calc_int_B_F()
        
            p.solve_PBE(t_vec = t)
            mu_pbe = p.calc_mom_t()
            print('### Initial dNdt..')
            # dNdt0=dNdt_2d(0,p.N[:,:,0].reshape(-1),p.V,p.V1_e,p.V3_e,p.F_M,NS, 0).reshape(NS,NS)
            # print(dNdt0)
        if MC:
            # Calculate multiple times (stochastic process)
            mu_tmp = []
            for l in range(N_MC):
                m = pop_mc(2)
                m.c = np.array([n0*v0,n0*v0])
                m.x = np.array([x,x])
                m.PGV = np.array(['mono','mono'])
                m.BETACALC = 1
                m.beta0 = beta0
                m.a0 = a0
                m.VERBOSE = VERBOSE
                m.savesteps = len(t)
                m.tA = t[-1]
                
                m.init_calc()
                
                m.solve_MC()
                mu_tmp.append(m.calc_mom_t())
                m_save.append(m)
            
            # Mean and STD of moments for all N_MC loops
            mu_mc = np.mean(mu_tmp,axis=0)
            if N_MC > 1: std_mu_mc = np.std(mu_tmp,ddof=1,axis=0)
        
        ### ANALYTICAL SOLUTION FROM KUMAR 2008: Eq. (40), (A.11)-(A.12) 
        if process_type == "agglomeration":
            n0_tot = 2*n0
            mu_as[0,0,:] = 2*n0_tot/(2+beta0*n0_tot*t)
            mu_as[1,0,:] = np.ones(t.shape)*c         
            mu_as[0,1,:] = np.ones(t.shape)*c
            mu_as[1,1,:] = c*c*beta0*n0_tot*t/n0_tot
            mu_as[2,0,:] = c*(v0+c*beta0*n0_tot*t/n0_tot) 
            mu_as[0,2,:] = c*(v0+c*beta0*n0_tot*t/n0_tot) 
        elif process_type == "breakage":
            for k in range(2):
                for l in range(2):
                    mu_as[k,l,:] = np.exp((2/((k+1)*(l+1))-1)*t)
        else:
            for k in range(2):
                for l in range(2):
                    mu_as[k,l,:] = np.ones(t.shape)*c  
        
    #%%% '3D_const_mono': 3D, constant kernel, monodisperse initial conditions
    elif CASE == '3D_const_mono':
        ### POPULATION BALANCE
        if PBE:
            p = pop_disc(3, disc=grid)
        
            p.NS = NS  
            p.process_type = process_type
            p.S = S
            p.COLEVAL = 3                           # Constant beta
            p.EFFEVAL = 2                           # Case for calculation of alpha
            p.CORR_BETA = beta0
            p.SIZEEVAL = 1
            p.R01, p.R02, p.R03 = x/2, x/2, x/2
            p.USE_PSD = False                  
            p.P1=0                                  # No breakage     
            p.N01, p.N02, p.N03 = n0, n0, n0
    
            p.calc_R()
            p.init_N()
            p.alpha_prim = np.ones(9)
            p.calc_F_M()
        
            p.solve_PBE(t_vec = t)
            mu_pbe = p.calc_mom_t()
        
        if MC:
            # Calculate multiple times (stochastic process)
            mu_tmp = []
            for l in range(N_MC):
                m = pop_mc(3)
                m.c = np.array([n0*v0,n0*v0,n0*v0])
                m.x = np.array([x,x,x])
                m.PGV = np.array(['mono','mono','mono'])
                m.BETACALC = 1
                m.beta0 = beta0
                m.a0 = a0
                m.VERBOSE = VERBOSE
                m.savesteps = len(t)
                m.tA = t[-1]
                
                m.init_calc()
                
                m.solve_MC()
                mu_tmp.append(m.calc_mom_t())
                m_save.append(m)
            
            # Mean and STD of moments for all N_MC loops
            mu_mc = np.mean(mu_tmp,axis=0)
            if N_MC > 1: std_mu_mc = np.std(mu_tmp,ddof=1,axis=0)
        
        ### ANALYTICAL SOLUTION ADAPTED FROM KUMAR 2008: Eq. (40), (A.11)-(A.12) 
        n0_tot = 2*n0
        mu_as[0,0,:] = 2*n0_tot/(2+beta0*n0_tot*t)
        mu_as[1,0,:] = np.ones(t.shape)*c     
        mu_as[0,1,:] = np.ones(t.shape)*c  
    
    #%%% '1D_sum_mono': 1D, sum kernel, monodisperse initial conditions
    elif CASE == '1D_sum_mono':
        ### POPULATION BALANCE
        if PBE:
            p = pop_disc(1, disc=grid)
        
            p.NS = NS  
            p.process_type = process_type
            p.S = S
            p.COLEVAL = 4                           # Sum kernel
            p.EFFEVAL = 2                           # Case for calculation of alpha
            p.CORR_BETA = beta0/v0
            p.SIZEEVAL = 1
            p.R01 = x/2
            p.USE_PSD = False                  
            p.P1=0                                  # No breakage     
            p.N01 = n0
              
            p.calc_R()
            p.init_N()
            p.alpha_prim = 1
            p.calc_F_M()
            p.calc_B_M()
        
            p.solve_PBE(t_vec = t)
            mu_pbe = p.calc_mom_t()
        
        if MC:
            # Calculate multiple times (stochastic process)
            mu_tmp = []
            for l in range(N_MC):
                m = pop_mc(1)
                m.c[0] = n0*v0
                m.x[0] = x
                m.PGV[0] = 'mono'
                m.BETACALC = 3
                m.beta0 = beta0/v0
                m.a0 = a0
                m.VERBOSE = VERBOSE
                m.savesteps = len(t)
                m.tA = t[-1]
                
                m.init_calc()
                
                m.solve_MC()
                mu_tmp.append(m.calc_mom_t())
                m_save.append(m)
            
            # Mean and STD of moments for all N_MC loops
            mu_mc = np.mean(mu_tmp,axis=0)
            if N_MC > 1: std_mu_mc = np.std(mu_tmp,ddof=1,axis=0)
        
        ### ANALYTICAL SOLUTION FROM KUMAR DISSERTATION A.11
        mu_as[0,0,:] = n0*np.exp(-beta0*n0*t) # without v0, therfore p.CORR_BETA also divided by v0
        mu_as[1,0,:] = np.ones(t.shape)*c
        phi = 1-np.exp(-beta0*n0*t)
        mu_as[2,0,:] = c*(v0+c*(2-phi)*phi/(n0*(1-phi)**2)) 
        
    #%%% '2D_sum_mono': 2D, sum kernel, monodisperse initial conditions
    elif CASE == '2D_sum_mono':
        ### POPULATION BALANCE
        if PBE:
            p = pop_disc(2, disc=grid)
        
            p.NS = NS  
            p.process_type = process_type
            p.S = S
            p.COLEVAL = 4                           # Sum kernel
            p.EFFEVAL = 2                           # Case for calculation of alpha
            p.CORR_BETA = beta0/v0
            p.SIZEEVAL = 1
            p.R01, p.R03 = x/2, x/2
            p.USE_PSD = False                  
            p.P1=0                                  # No breakage     
            p.N01, p.N03 = n0, n0
    
            p.calc_R()
            p.init_N()
            p.alpha_prim = np.ones(4)
            p.calc_F_M()
        
            p.solve_PBE(t_vec = t)
            mu_pbe = p.calc_mom_t()
        
        if MC:
            # Calculate multiple times (stochastic process)
            mu_tmp = []
            for l in range(N_MC):
                m = pop_mc(2)
                m.c = np.array([n0*v0,n0*v0])
                m.x = np.array([x,x])
                m.PGV = np.array(['mono','mono'])
                m.BETACALC = 3
                m.beta0 = beta0/v0              
                m.a0 = a0
                m.VERBOSE = VERBOSE
                m.savesteps = len(t)
                m.tA = t[-1]
                
                m.init_calc()
                
                m.solve_MC()
                mu_tmp.append(m.calc_mom_t())
                m_save.append(m)
            
            # Mean and STD of moments for all N_MC loops
            mu_mc = np.mean(mu_tmp,axis=0)
            if N_MC > 1: std_mu_mc = np.std(mu_tmp,ddof=1,axis=0)
        
        ### ANALYTICAL SOLUTION FROM KUMAR DISSERTATION A.11
        n0_tot = 2*n0
        mu_as[0,0,:] = n0_tot*np.exp(-beta0*n0_tot*t) # without v0, therfore p.CORR_BETA also divided by v0
        mu_as[1,0,:] = np.ones(t.shape)*c
        mu_as[0,1,:] = np.ones(t.shape)*c
        phi = 1-np.exp(-beta0*n0_tot*t)        
        mu_as[1,1,:] = c*c*(2-phi)*phi/(n0_tot*(1-phi)**2)
        mu_as[2,0,:] = c*(v0+c*(2-phi)*phi/(n0_tot*(1-phi)**2)) 
    
    #%%% '2D_sum_mono_ccm': 2D, sum kernel, monodisperse initial conditions, aplha from CCM
    elif CASE == '2D_sum_mono_ccm':
        ### POPULATION BALANCE
        if PBE:
            p = pop_disc(2, disc=grid)
        
            p.NS = NS 
            p.process_type = process_type
            p.S = S
            p.COLEVAL = 4                           # Sum kernel
            p.EFFEVAL = 2                           # Case for calculation of alpha
            p.CORR_BETA = beta0/v0
            p.SIZEEVAL = 1
            p.R01, p.R03 = x/2, x/2
            p.USE_PSD = False                  
            p.P1=0                                  # No breakage     
            p.N01, p.N03 = n0, n0
    
            p.calc_R()
            p.init_N()
            p.alpha_prim = alpha_pbe
            p.calc_F_M()
        
            p.solve_PBE(t_vec = t)
            mu_pbe = p.calc_mom_t()
        
        if MC:          
            if MULTI_INTERNAL:
                m = pop_mc(2)
                m.c = np.array([n0*v0,n0*v0])
                m.x = np.array([x,x])
                m.PGV = np.array(['mono','mono'])
                m.BETACALC = 3
                m.ALPHACALC = 2
                m.beta0 = beta0/v0              
                m.a0 = a0
                m.VERBOSE = VERBOSE
                m.alpha_prim = alpha_mc
                m.savesteps = len(t)
                m.tA = t[-1]
                
                m.init_calc()
                
                m.solve_MC_N(N_MC)
                mu_mc = m.calc_mom_t()
                m_save.append(m)
            else:                
                # Calculate multiple times (stochastic process)
                mu_tmp = []
                for l in range(N_MC):
                    m = pop_mc(2)
                    m.c = np.array([n0*v0,n0*v0])
                    m.x = np.array([x,x])
                    m.PGV = np.array(['mono','mono'])
                    m.BETACALC = 3
                    m.ALPHACALC = 2
                    m.beta0 = beta0/v0              
                    m.a0 = a0
                    m.VERBOSE = VERBOSE
                    m.alpha_prim = alpha_mc
                    m.savesteps = len(t)
                    m.tA = t[-1]
                    
                    m.init_calc()
                    
                    m.solve_MC()
                    mu_tmp.append(m.calc_mom_t())
                    m_save.append(m)
                
                # Mean and STD of moments for all N_MC loops
                mu_mc = np.mean(mu_tmp,axis=0)
                if N_MC > 1: std_mu_mc = np.std(mu_tmp,ddof=1,axis=0)
        
        ### ANALYTICAL SOLUTION FOR SPECIAL CASES OF ALPHA_PRIM KNOWN
        if all(alpha_pbe == np.ones(4)):
            n0_tot = 2*n0
            mu_as[0,0,:] = n0_tot*np.exp(-beta0*n0_tot*t) # without v0, therfore p.CORR_BETA also divided by v0
            mu_as[1,0,:] = np.ones(t.shape)*c
            mu_as[0,1,:] = np.ones(t.shape)*c
            phi = 1-np.exp(-beta0*n0_tot*t)        
            mu_as[1,1,:] = c*c*(2-phi)*phi/(n0_tot*(1-phi)**2)
            mu_as[2,0,:] = c*(v0+c*(2-phi)*phi/(n0_tot*(1-phi)**2)) 
        elif all(alpha_pbe == np.array([1,0,0,0])):
            # mu_as[0,0,:] = n0+2*n0/(2+beta0*n0*t)
            mu_as[0,0,:] = 2*2*n0/(2+beta0*n0*t)
            mu_as[1,0,:] = np.ones(t.shape)*c 
        else:    
            mu_as = None
            
    #%%% '2D_ortho_mono': 2D, ortho kernel, monodisperse initial conditions, alpha = 1
    elif CASE == '2D_ortho_mono':
        beta_fac = 0.5                              # Scale all betas in this case to make it slower
        ### POPULATION BALANCE
        if PBE:
            p = pop_disc(2, disc=grid)
        
            p.NS = NS  
            p.process_type = process_type
            p.S = S
            p.COLEVAL = 1                           # Constant beta
            p.EFFEVAL = 2                           # Case for calculation of alpha
            p.CORR_BETA = beta_fac*beta0/v0             
            p.SIZEEVAL = 1
            p.R01, p.R03 = x/2, x/2
            p.USE_PSD = False                  
            p.P1=0                                  # No breakage     
            p.N01, p.N03 = n0, n0
    
            p.calc_R()
            p.init_N()
            p.alpha_prim = np.ones(4)
            p.calc_F_M()
        
            p.solve_PBE(t_vec = t)
            mu_pbe = p.calc_mom_t()
        
        if MC:
            # Calculate multiple times (stochastic process)
            mu_tmp = []
            for l in range(N_MC):
                m = pop_mc(2)
                m.c = np.array([n0*v0,n0*v0])
                m.x = np.array([x,x])
                m.PGV = np.array(['mono','mono'])
                m.BETACALC = 2
                m.beta0 = beta_fac*beta0/v0              
                m.a0 = a0
                m.VERBOSE = VERBOSE
                m.savesteps = len(t)
                m.tA = t[-1]
                
                m.init_calc()
                
                m.solve_MC()
                mu_tmp.append(m.calc_mom_t())
                m_save.append(m)
            
            # Mean and STD of moments for all N_MC loops
            mu_mc = np.mean(mu_tmp,axis=0)
            if N_MC > 1: std_mu_mc = np.std(mu_tmp,ddof=1,axis=0)
        
        ### NO ANALYTICAL SOLUTION AVAILABLE
        mu_as = None
        
    #%%% '2D_ortho_mono_ccm': 2D, ortho kernel, monodisperse initial conditions, alpha from CCM
    elif CASE == '2D_ortho_mono_ccm':
        beta_fac = 0.5                              # Scale all betas in this case to make it slower
        ### POPULATION BALANCE
        if PBE:
            p = pop_disc(2, disc=grid)
        
            p.NS = NS  
            p.S = S
            p.COLEVAL = 1                           # Constant beta
            p.EFFEVAL = 2                           # Case for calculation of alpha
            p.CORR_BETA = beta_fac*beta0/v0             
            p.SIZEEVAL = 1
            p.R01, p.R03 = x/2, x/2
            p.USE_PSD = False                  
            p.P1=0                                  # No breakage     
            p.N01, p.N03 = n0, n0
    
            p.calc_R()
            p.init_N()
            p.alpha_prim = alpha_pbe
            p.calc_F_M()
        
            p.solve_PBE(t_vec = t)
            mu_pbe = p.calc_mom_t()
        
        if MC:
            # Calculate multiple times (stochastic process)
            mu_tmp = []
            for l in range(N_MC):
                m = pop_mc(2)
                m.c = np.array([n0*v0,n0*v0])
                m.x = np.array([x,x])
                m.PGV = np.array(['mono','mono'])
                m.BETACALC = 2
                m.ALPHACALC = 2
                m.beta0 = beta_fac*beta0/v0              
                m.a0 = a0
                m.VERBOSE = VERBOSE
                m.alpha_prim = alpha_mc
                m.savesteps = len(t)
                m.tA = t[-1]
                
                m.init_calc()
                
                m.solve_MC()
                mu_tmp.append(m.calc_mom_t())
                m_save.append(m)
            
            # Mean and STD of moments for all N_MC loops
            mu_mc = np.mean(mu_tmp,axis=0)
            if N_MC > 1: std_mu_mc = np.std(mu_tmp,ddof=1,axis=0)
        
        ### NO ANALYTICAL SOLUTION AVAILABLE
        mu_as = None  
        
    else:
        print('Provided Case not coded yet')

    return mu_as, mu_pbe, mu_mc, std_mu_mc, p, m, mu_tmp, m_save
    

#%% FUNCTIONS
def init_plot(default = False, size = 'half', extra = False, mrksize = 5):
    
    if size == 'full':
        pt.plot_init(scl_a4=1, page_lnewdth_cm=13.858, figsze=[12.8,4.8*(4/3)],lnewdth=0.8,
                     mrksze=mrksize,use_locale=True, fontsize=9, labelfontsize=9, tickfontsize=8)
        if extra:
            pt.plot_init(scl_a4=1, page_lnewdth_cm=18.486, figsze=[12.8*(18.486/13.858),4.8*(4/3)],lnewdth=0.8,
                     mrksze=mrksize,use_locale=True, fontsize=9, labelfontsize=9, tickfontsize=8)
    if size == 'half':
        pt.plot_init(scl_a4=2, page_lnewdth_cm=13.858, figsze=[6.4,4.8*(4/3)],lnewdth=0.8,
                     mrksze=mrksize,use_locale=True, fontsize=9, labelfontsize=9, tickfontsize=8)
        if extra:
            pt.plot_init(scl_a4=2, page_lnewdth_cm=18.486, figsze=[6.4*(18.486/13.858),4.8*(4/3)],lnewdth=0.8,
                     mrksze=mrksize,use_locale=True, fontsize=9, labelfontsize=9, tickfontsize=8)
    if default:
            pt.plot_init(scl_a4=2, page_lnewdth_cm=13.858, figsze=[6.4,4.8*(4/3)],lnewdth=0.8,
                     mrksze=mrksize,use_locale=True, fontsize=9, labelfontsize=9, tickfontsize=8)
            
            
def plot_moment_t(mu_as=None, mu_pbe=None, mu_mc=None, std_mu_mc=None, i=0, j=0, fig=None, ax=None, label=None,
                  labelpos='sw',t_mod=None, rel=False, alpha=1):
    
    if fig is None or ax is None:
        fig=plt.figure()    
        ax=fig.add_subplot(1,1,1)   
    
    if t_mod is None:
        tp = t
    else:
        tp = t_mod
        
    if rel:
        ylbl = 'Relative Moment $\mu_{' + f'{i}{j}' + '}\,/\,\mu_{' + f'{i}{j}' + '}(0)$ / $-$'
    else:
        ylbl = 'Moment $\mu_{' + f'{i}{j}' + '}$ / '+'$m^{3\cdot'+str(i+j)+'}$'
        
    if mu_as is not None:
        if rel: mu_as[i,j,:] = mu_as[i,j,:]/mu_as[i,j,0]
        ax, fig = pt.plot_data(tp,mu_as[i,j,:], fig=fig, ax=ax,
                               xlbl='Agglomeration time $t_\mathrm{A}$ / $s$',
                               ylbl=ylbl, alpha=alpha,
                               lbl='Analytical Solution',clr='k',mrk='o')

    if mu_mc is not None:
        if rel: 
            if std_mu_mc is not None:
                std_mu_mc[i,j,:] = std_mu_mc[i,j,:]/mu_mc[i,j,0]
            mu_mc[i,j,:] = mu_mc[i,j,:]/mu_mc[i,j,0]
        
        if std_mu_mc is not None:
            ax, fig = pt.plot_data(tp,mu_mc[i,j,:], err=std_mu_mc[i,j,:], fig=fig, ax=ax,
                                   xlbl='Agglomeration time $t_\mathrm{A}$ / $s$',
                                   ylbl=ylbl, lbl='MC, $N_{\mathrm{MC}}='+str(N_MC)+'$',
                                   clr=c_KIT_red,mrk='s', alpha=alpha, mrkedgecolor='k')
        else:
            ax, fig = pt.plot_data(tp,mu_mc[i,j,:], fig=fig, ax=ax,
                                   xlbl='Agglomeration time $t_\mathrm{A}$ / $s$',
                                   ylbl=ylbl, lbl='MC, $N_{\mathrm{MC}}='+str(N_MC)+'$',
                                   clr=c_KIT_red,mrk='s', alpha=alpha, mrkedgecolor='k')
        
    if mu_pbe is not None:
        if rel: mu_pbe[i,j,:] = mu_pbe[i,j,:]/mu_pbe[i,j,0]
        ax, fig = pt.plot_data(tp,mu_pbe[i,j,:], fig=fig, ax=ax,
                               xlbl='Agglomeration time $t_\mathrm{A}$ / $s$',
                               ylbl=ylbl, lbl='dPBE, $N_{\mathrm{S}}='+str(NS)+'$',
                               clr=c_KIT_green,mrk='^', alpha=alpha, mrkedgecolor='k')
    
    # if std_mu_mc is not None:
    #     ax.errorbar(tp,mu_mc[i,j,:],yerr=std_mu_mc[i,j,:],fmt='none',color=c_KIT_red,
    #                 capsize=plt.rcParams['lines.markersize']-2,alpha=alpha,zorder=0, mec ='k')
        
    # Adjust y scale in case of first moment
    if i+j == 1 and mu_as is not None and mu_pbe is not None and mu_mc is not None:
        ax.set_ylim([np.min([mu_pbe[i,j,:]])*0.9,np.max([mu_pbe[i,j,:]])*1.1])
        # ax.set_ylim([np.min([mu_pbe[i,j,:],mu_as[i,j,:],mu_mc[i,j,:]])*0.9,
        #              np.max([mu_pbe[i,j,:],mu_as[i,j,:],mu_mc[i,j,:]])*1.1])
    elif i+j == 1 and mu_as is not None:
        ax.set_ylim([np.min([mu_as[i,j,:]])*0.9,np.max([mu_as[i,j,:]])*1.1])
    elif i+j == 1 and mu_pbe is not None:
        ax.set_ylim([np.min([mu_pbe[i,j,:]])*0.9,np.max([mu_pbe[i,j,:]])*1.1])
    elif i+j == 1 and mu_mc is not None:
        ax.set_ylim([np.min([mu_mc[i,j,:]])*0.9,np.max([mu_mc[i,j,:]])*1.1])
    
    if label is not None:
        if labelpos == 'se':
            ax.text(0.98,0.02,label,transform=ax.transAxes,horizontalalignment='right',
                    verticalalignment='bottom',bbox=dict(alpha=0.8,facecolor='w', edgecolor='none',pad=1.2))
        else:                
            ax.text(0.02,0.02,label,transform=ax.transAxes,horizontalalignment='left',
                    verticalalignment='bottom',bbox=dict(alpha=0.8,facecolor='w', edgecolor='none',pad=1.2))
        
    # For moments larger 0 and 1 scale y axis logarithmically
    if i+j >= 2:
        ax.set_yscale('log')
    
    
    ax.yaxis.set_major_formatter(ScalarFormatter())
    
    ax.grid('minor')
    plt.tight_layout()   
    
    return ax, fig

def add_moment_t(mu, fig, ax, i=0, j=0, lbl=None, t_mod=None, rel=False, alpha=1):
    
    if t_mod is None:
        tp = t
    else:
        tp = t_mod
        
    if lbl is None:
        lbl = 'dPBE, $N_{\mathrm{S}}='+str(NS)+'$'
    
    if rel: mu[i,j,:] = mu[i,j,:]/mu[i,j,0]
    ax, fig = pt.plot_data(tp,mu[i,j,:], fig=fig, ax=ax, alpha=alpha,
                           lbl=lbl,clr=c_KIT_green,mrk='v', mrkedgecolor='k')
    
    return ax, fig

def plot_Q3(m_save, p, p2=None, alpha=1, label=None, Q3_grid=np.linspace(0,1,21)):
    x_mc_full = np.zeros((len(Q3_grid),len(m_save)))
    for i in range(len(m_save)):
        x_mc_full[:,i], _, Q3_mc, _, _, _ = m_save[i].return_distribution(t=-1, Q3_grid=Q3_grid)
    
    x_mc = np.mean(x_mc_full, axis=1)
    x_mc_std = np.std(x_mc_full, axis=1)
    
    x_pbe, Q3_pbe= p.return_distribution(t=-1,flag='x_uni,q3')

    fig=plt.figure()    
    ax=fig.add_subplot(1,1,1)   
    
    ax, fig = pt.plot_data(x_mc, Q3_mc, err=x_mc_std, fig=fig, ax=ax,
                           xlbl='Equivalent Diameter $d$ / $\mathrm{\mu m}$',
                           ylbl='Cumulative Distribution $Q_3$ / $-$', alpha=alpha,
                           clr=c_KIT_red,mrk='s', mrkedgecolor='k', err_ax='x',
                           lbl='MC, $N_{\mathrm{MC}}='+str(N_MC)+'$')
    
    ax, fig = pt.plot_data(x_pbe, Q3_pbe, fig=fig, ax=ax,
                           xlbl='Equivalent Diameter $d$ / $\mathrm{\mu m}$',
                           ylbl='Cumulative Distribution $Q_3$ / $-$', alpha=alpha,
                           clr=c_KIT_green,mrk='^', mrkedgecolor='k', 
                           lbl='dPBE, $N_{\mathrm{S}}='+str(p.NS)+'$')
    
    if p2 is not None:
        x_pbe2, Q3_pbe2 = p2.return_distribution(t=-1,flag='x_uni,q3')
        ax, fig = pt.plot_data(x_pbe2, Q3_pbe2, fig=fig, ax=ax,
                               xlbl='Equivalent Diameter $d$ / $\mathrm{\mu m}$',
                               ylbl='Cumulative Distribution $Q_3$ / $-$', alpha=alpha,
                               clr=c_KIT_green,mrk='v', mrkedgecolor='k', 
                               lbl='dPBE, $N_{\mathrm{S}}='+str(p2.NS)+'$')
    
    # Add invisible point to adjust x scaling
    ax.plot(min(x_pbe)*10,0.5,alpha=0)
    ax.set_xlim([min(x_pbe)*0.95,ax.get_xlim()[1]])
    
    if label is not None:
        ax.text(0.98,0.02,label,transform=ax.transAxes,horizontalalignment='right',
                verticalalignment='bottom',bbox=dict(alpha=0.8,facecolor='w', edgecolor='none',pad=1.2))
    
    ax.set_xscale('log')
    ax.grid('minor')
    plt.tight_layout()  
    
    return ax, fig, x_mc, x_mc_std, x_mc_full

#%% MAIN    
if __name__ == "__main__":
    ### Plot parameters
    EXP = True
    #EXPPTH = 'export/'
    EXPPTH = 'C:/Users/px2030/Code/PSD_opt/general_scripts/temp/'
    EXPRAW = False
    REL = True
    
    ### Define calculation case
    # '1D_const_mono': 1D, constant kernel, monodisperse initial conditions
    # '2D_const_mono': 2D, constant kernel, monodisperse initial conditions
    # '3D_const_mono': 3D, constant kernel, monodisperse initial conditions
    # '1D_sum_mono': 1D, sum kernel, monodisperse initial conditions
    # '2D_sum_mono': 2D, sum kernel, monodisperse initial conditions
    # '2D_sum_mono_ccm': 2D, sum kernel, monodisperse initial conditions, aplha from CCM
    # '2D_ortho_mono': 2D, ortho kernel, monodisperse initial conditions, alpha = 1
    # '2D_ortho_mono': 2D, ortho kernel, monodisperse initial conditions, alpha from CCM
    CASE = '1D_const_mono'
    # CASE = '2D_const_mono'
    #CASE = '3D_const_mono'
    # CASE = '1D_sum_mono'
    # CASE = '2D_sum_mono'
    #CASE = '2D_sum_mono_ccm'
    #CASE = '2D_ortho_mono'
    #CASE = '2D_ortho_mono_ccm'
    
    ### General parameters
    t = np.arange(0, 601, 60, dtype=float)     # Time array [s]
    c = 0.1e-2*1e-2                 # Volume concentration [-]
    # v0 = 1e-9
    x = 1e-3                       # Particle diameter [m]
    # x = (v0*6/math.pi)**(1/3)
    beta0 = 1e-16                   # Collision frequency parameter [m^3/s]
    n0 = 3*c/(4*math.pi*(x/2)**3)   # Total number concentration of primary particles
    # n0 = 1                        # validation for pure breakage
    v0 = 4*math.pi*(x/2)**3/3       # Volume of a primary particle
    MULTI_INTERNAL = False
    
    ### PBE Parameters
    grid = 'geo'
    NS = 15
    # NS2 = 15
    #NS2 = 50
    process_type = "breakage"
    
    S = 4
    # alpha_pbe = np.array([1,0.2,0.2,0])
    alpha_pbe = np.array([1,1,1,1])
    # alpha_pbe = np.array([1,0,0,0])
    
    ### MC Parameters
    a0 = 200
    N_MC = 5
    VERBOSE = True    
    alpha_mc = np.reshape(alpha_pbe,(2,2))
    
    EFFEVAL=2       #dPB
    COLEVAL=3       #dPB
    BREAKFVAL = 2
    BREAKRVAL = 2
    pl_v = 0.5   ## number of fragments
    pl_q = 1   ## parameter describes the breakage type
    pl_P1 = 1e-4 
    pl_P2 = 0.5
    G = 2.3

    mu_as, mu_pbe, mu_mc, std_mu_mc, p, m, mu_mc_reps, m_save  = calculate_case(CASE,MC=False)
    
    #%% PLOTS
    # pt.close()
    init_plot(size = 'half', extra = True, mrksize=4)
    
    ALPHA = 0.7
    
    ax1, fig1 = plot_moment_t(mu_as, mu_pbe, mu_mc, std_mu_mc = std_mu_mc, i=0, j=0, label='(a)', rel=REL, alpha = ALPHA)
    ax2, fig2 = plot_moment_t(mu_as, mu_pbe, mu_mc, std_mu_mc = std_mu_mc, i=1, j=0, label='(b)', rel=REL, alpha = ALPHA)
    ax4, fig4 = plot_moment_t(mu_as, mu_pbe, mu_mc, std_mu_mc = std_mu_mc, i=2, j=0, label='(d)',
                              labelpos='se', rel=REL, alpha = ALPHA)
    if p.dim == 2:
        ax3, fig3 = plot_moment_t(mu_as[:,:,1:], mu_pbe[:,:,1:], mu_mc[:,:,1:], std_mu_mc = std_mu_mc[:,:,1:], 
                                  i=1, j=1, t_mod=t[1:], label='(c)',labelpos='se', rel=REL, alpha = ALPHA)
        
    if NS2 is not None:
        NS = NS2
        _, mu_pbe2, _, _, p2, _, _, _  = calculate_case(CASE)
        ax1, fig1 = add_moment_t(mu_pbe2, fig1, ax1, i=0, j=0, rel=REL, alpha = ALPHA)
        ax2, fig2 = add_moment_t(mu_pbe2, fig2, ax2, i=1, j=0, rel=REL, alpha = ALPHA)
        ax4, fig4 = add_moment_t(mu_pbe2, fig4, ax4, i=2, j=0, rel=REL, alpha = ALPHA)
        if p.dim == 2:
            ax3, fig3 = add_moment_t(mu_pbe2[:,:,1:], fig3, ax3, i=1, j=1, t_mod=t[1:], rel=REL, alpha = ALPHA)
            
    ax2.legend().remove()
    if p.dim == 2: ax3.legend().remove()
    ax4.legend().remove()
    
    # ax5, fig5, x_mc, x_mc_std, x_mc_full = plot_Q3(m_save, p, p2, alpha=ALPHA, label='(d)')
    # ax5.legend().remove()
    
    # if EXP:
    #     fig1.savefig(EXPPTH+f'{CASE}_MU00.png',dpi=300)
    #     fig2.savefig(EXPPTH+f'{CASE}_MU10.png',dpi=300)
    #     fig4.savefig(EXPPTH+f'{CASE}_MU20.png',dpi=300)
    #     if p.dim == 2:
    #         fig3.savefig(EXPPTH+f'{CASE}_MU11.png',dpi=300)   
    #     fig5.savefig(EXPPTH+f'{CASE}_Q3.pdf')
            
    # if EXPRAW:
    #     from datetime import datetime
    #     current_time = datetime.now()
    #     formatted_time = current_time.strftime('%Y%m%d_%H%M%S')
    #     np.save(EXPPTH+'raw/raw_data_valid_'+formatted_time+'.npy',
    #             {'mu_as':mu_as, 'mu_pbe':mu_pbe, 'mu_mc':mu_mc, 
    #              'std_mu_mc':std_mu_mc, 'p':p, 'm':m, 
    #              'mu_mc_reps':mu_mc_reps, 'm_save':m_save})
        


