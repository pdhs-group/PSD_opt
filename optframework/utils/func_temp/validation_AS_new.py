# -*- coding: utf-8 -*-
"""
Validate discrete PBE with analytical solutions for various cases

@author: xy0264
"""
# %% IMPORTS
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import numpy as np
import math
import optframework.utils.plotter.plotter as pt
from optframework.utils.plotter.KIT_cmap import c_KIT_green, c_KIT_red, c_KIT_blue
from optframework.dpbe.dpbe_base import DPBESolver as pop_disc
from optframework.mcpbe.mcpbe import population_MC as pop_mc 

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
            p = pop_disc(1, disc=grid, load_attr=False)
            
            p.process_type = process_type
            p.BREAKFVAL = 2
            p.BREAKRVAL = 2
            
            p.NS = NS  
            p.S = S
            p.COLEVAL = 3                          # Constant beta
            p.EFFEVAL = 2                           # Case for calculation of alpha
            p.CORR_BETA = beta0
            p.SIZEEVAL = 1
            p.R01 = x/2
            p.USE_PSD = False                      
            # p.N01 = n0
              
            p.calc_R()
            p.init_N(reset_N=True, N01=n0)
            p.alpha_prim = alpha_pbe[0]
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
        else:
            print("not yet coded")
    
    #%%% '2D_const_mono': 2D, constant kernel, monodisperse initial conditions
    elif CASE == '2D_const_mono':
        ### POPULATION BALANCE
        if PBE:
            p = pop_disc(2, disc=grid, load_attr=False)
            
            p.process_type = process_type
            p.BREAKFVAL = 2
            p.BREAKRVAL = 1
            
            p.NS = NS  
            p.S = S
            p.COLEVAL = 3                           # Constant beta
            p.EFFEVAL = 2                           # Case for calculation of alpha
            p.CORR_BETA = beta0
            p.SIZEEVAL = 1
            p.R01, p.R03 = x/2, x/2
            p.USE_PSD = False                  
            # p.N01, p.N03 = n0, n0
    
            p.calc_R()
            p.init_N(reset_N=True, N01=n0, N03=n0)
            p.alpha_prim = np.ones(4)
            p.calc_F_M()
            p.calc_B_R()
            p.calc_int_B_F()
        
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
            
        ### See Leong-Table 1.
        elif process_type == "breakage":
            for k in range(2):
                for l in range(2):
                    mu_as[k,l,:] = (p.V1[-1])**k*(p.V3[-1])**l*np.exp((2/((k+1)*(l+1))-1)*t)
        else:
            for k in range(2):
                for l in range(2):
                    mu_as[k,l,:] = np.ones(t.shape)*c  
    
    #%%% '1D_sum_mono': 1D, sum kernel, monodisperse initial conditions
    elif CASE == '1D_sum_mono':
        ### POPULATION BALANCE
        if PBE:
            p = pop_disc(1, disc=grid, load_attr=False)
        
            p.NS = NS  
            p.process_type = process_type
            p.S = S
            p.BREAKFVAL = 2
            p.BREAKRVAL = 2
            p.COLEVAL = 4                           # Sum kernel
            p.EFFEVAL = 2                           # Case for calculation of alpha
            p.CORR_BETA = beta0/v0
            p.SIZEEVAL = 1
            p.R01 = x/2
            p.USE_PSD = False                    
            # p.N01 = n0
              
            p.calc_R()
            p.init_N(reset_N=True, N01=n0)
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
        
        if process_type == "agglomeration":
            ### ANALYTICAL SOLUTION FROM KUMAR DISSERTATION A.11
            mu_as[0,0,:] = n0*np.exp(-beta0*n0*t) # without v0, therfore p.CORR_BETA also divided by v0
            mu_as[1,0,:] = np.ones(t.shape)*c
            phi = 1-np.exp(-beta0*n0*t)
            mu_as[2,0,:] = c*(v0+c*(2-phi)*phi/(n0*(1-phi)**2)) 
        elif process_type == "breakage":
            if grid == 'uni':
                V = p.V
            else:
                V = p.V_e
            # see Kumar Dissertation A.1
            N_as = np.zeros((NS,len(t)))
            V_sum = np.zeros((NS,len(t)))
            for i in range(NS):
                for j in range(len(t)):
                    if i != NS-1:
                        N_as[i,j] = (-(t[j]*p.V[-1]+1)+t[j]*V[i+1])*np.exp(-V[i+1]*t[j])-\
                            (-(t[j]*p.V[-1]+1)+t[j]*V[i])*np.exp(-V[i]*t[j])
                    else:
                        N_as[i,j] = (-(t[j]*p.V[-1]+1)+t[j]*p.V[i])*np.exp(-p.V[i]*t[j])-\
                            (-(t[j]*p.V[-1]+1)+t[j]*V[i])*np.exp(-V[i]*t[j]) + \
                            (np.exp(-t[j]*p.V[i]))
                    V_sum[i,j] = N_as[i,j] * p.V[i]
            mu_as[0,0,:] = N_as.sum(axis=0)
            mu_as[1,0,:] = np.ones(t.shape)*c 
        else:
            mu_as[0,0,:] = np.ones(t.shape)*c 
            mu_as[1,0,:] = np.ones(t.shape)*c 
        
    #%%% '2D_sum_mono': 2D, sum kernel, monodisperse initial conditions
    elif CASE == '2D_sum_mono':
        ### POPULATION BALANCE
        if PBE:
            p = pop_disc(2, disc=grid, t_vec=t, load_attr=False)
        
            p.NS = NS  
            p.process_type = process_type
            p.S = S
            p.BREAKFVAL = 2
            p.BREAKRVAL = 2
            p.COLEVAL = 4                           # Sum kernel
            p.EFFEVAL = 2                           # Case for calculation of alpha
            p.CORR_BETA = beta0/v0
            p.SIZEEVAL = 1
            p.R01, p.R03 = x/2, x/2
            p.USE_PSD = False                    
            # p.N01, p.N03 = n0, n0
    
            p.calc_R()
            p.init_N(reset_N=True, N01=n0, N03=n0)
            p.alpha_prim = np.ones(4)
            p.calc_F_M()
            p.calc_B_R()
            p.calc_int_B_F()
        
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
        
        if process_type == "agglomeration":
            ### ANALYTICAL SOLUTION FROM KUMAR DISSERTATION A.11
            n0_tot = 2*n0
            mu_as[0,0,:] = n0_tot*np.exp(-beta0*n0_tot*t) # without v0, therfore p.CORR_BETA also divided by v0
            mu_as[1,0,:] = np.ones(t.shape)*c
            mu_as[0,1,:] = np.ones(t.shape)*c
            phi = 1-np.exp(-beta0*n0_tot*t)        
            mu_as[1,1,:] = c*c*(2-phi)*phi/(n0_tot*(1-phi)**2)
            mu_as[2,0,:] = c*(v0+c*(2-phi)*phi/(n0_tot*(1-phi)**2)) 
        elif process_type == "breakage":
            ### See Leong-Table 1.
            mu_as[0,1,:] = p.V1[-1]
            mu_as[1,0,:] = p.V3[-1]
            mu_as[0,0,:] = 1 + p.V[-1,-1]*t
            # for k in range(2):
            #     for l in range(2):
            #         mu_as[k,l,:] = np.exp((2/((k+1)*(l+1))-1)*t)
        else:
            for k in range(2):
                for l in range(2):
                    mu_as[k,l,:] = np.ones(t.shape)*c  
        
        

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
    
    x_pbe, Q3_pbe= p.return_distribution(t=-1,flag='x_uni,qx')

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
        x_pbe2, Q3_pbe2 = p2.return_distribution(t=-1,flag='x_uni,qx')
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
    REL = True
    
    ### Define calculation case
    # '1D_const_mono': 1D, constant kernel, monodisperse initial conditions
    # '2D_const_mono': 2D, constant kernel, monodisperse initial conditions
    # '1D_sum_mono': 1D, sum kernel, monodisperse initial conditions
    # '2D_sum_mono': 2D, sum kernel, monodisperse initial conditions
    # '2D_sum_mono_ccm': 2D, sum kernel, monodisperse initial conditions, aplha from CCM
    # CASE = '1D_const_mono'
    # CASE = '2D_const_mono'
    # CASE = '1D_sum_mono'
    CASE = '2D_sum_mono'
    
    ### General parameters
    t = np.arange(0, 100, 5, dtype=float)     # Time array [s]
    c = 1                # Volume concentration [-]
    x = 2e-1            # Particle diameter [m]
    n0 = 3*c/(4*math.pi*(x/2)**3)   # Total number concentration of primary particles
    
    ### PBE Parameters
    grid = 'geo'
    S = 4
    NS = 10
    NS2 = None
    #NS2 = 50
    process_type = "breakage"
    
    beta0 = 1e-3                 # Collision frequency parameter [m^3/s]
    v0 = 4*math.pi*(x/2)**3/3*(1+S)/2
    alpha_pbe = np.array([1,1,1,1])
    
    ### MC Parameters
    a0 = 200
    N_MC = 5
    VERBOSE = True    
    alpha_mc = np.reshape(alpha_pbe,(2,2))
    

    mu_as, mu_pbe, mu_mc, std_mu_mc, p, m, mu_mc_reps, m_save  = calculate_case(CASE,MC=True)
    
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
            
    # ax2.legend().remove()
    # if p.dim == 2: ax3.legend().remove()
    # ax4.legend().remove()
    
    # ax5, fig5, x_mc, x_mc_std, x_mc_full = plot_Q3(m_save, p, p2, alpha=ALPHA, label='(d)')
    # ax5.legend().remove()
