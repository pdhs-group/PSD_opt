# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 08:53:21 2025

@author: px2030
"""
import os
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from optframework.pbe import DPBESolver, ExtruderPBESolver
from optframework.pbe import MCPBESolver
from optframework.pbm import PBMSolver
import optframework.utils.plotter.plotter as pt
from optframework.utils.plotter.KIT_cmap import c_KIT_green, c_KIT_red, c_KIT_blue

MIN = 1e-20

class PBEValidation():
    def __init__(self, dim, grid, NS, S, kernel, process,
                 c=1, x=2e-6, beta0=1e-2, t=None, NC=3, Extruder=False,
                 mom_n_order=2, mom_n_add=10, use_psd=False, dist_path=None):
        self.x = x
        self.beta0 = beta0
        self.kernel = kernel
        self.Extruder = Extruder
        self.NC = NC
        self.c = c
        self.n0 = 3*c/(4*math.pi*(x/2)**3)
        self.mom_order = 2*mom_n_order
        self.use_psd = use_psd
        self.dist_path = dist_path
        ## Check if the psd file is available
        if self.use_psd:
            if self.dist_path is None:
                raise ValueError("The full path to initial PSD file must be provided!")
            elif not os.path.exists(self.dist_path):
                raise FileNotFoundError(f"The provided initial PSD file not found at: {self.dist_path}")
            init_psd = np.load(dist_path, allow_pickle=True).item()
            ## The particle distribution created by full_psd follows the volume normal distribution, 
            ## so the resulting number density distribution is usually concentrated on the left side of the coordinate axis.
            ## If the original coordinates are used directly, 
            ## there will be only a large number of small particles at the initial moment, 
            ## which may cause large errors in the calculation of dPBE under breakage case. 
            ## The scaling factor here is equivalent to translating the entire coordinate axis, 
            ## making the peak of the number density distribution closer to the middle position, 
            ## and will not affect the absolute value of the calculation result.
            self.x = init_psd['r0_001'] * 2 * 1e-1
        self.init_pbe(NS, S, dim, t, grid, process)
        self.init_mcpbe(dim, t, process)
        self.init_pbm(dim, t, process, mom_n_order, mom_n_add)

    def init_pbe(self, NS, S, dim, t, grid, process):
        if not self.Extruder:
            self.p = DPBESolver(dim=dim, t_vec=t, load_attr=False,f=grid)
        else:
            self.p = ExtruderPBESolver(dim=dim, NC=self.NC, t_vec=t, load_attr=False,disc=grid)
        self.p.NS = NS
        self.p.S = S
        self.p.USE_PSD = self.use_psd
        self.p.R01, self.p.R03 = self.x/2, self.x/2
        self.p.DIST1, self.p.DIST1 = self.dist_path, self.dist_path
        self.p.alpha_prim = np.ones(dim**2)
        self.p.G = 1.0
        self.p.process_type = process
        self.p.calc_R()
        self.p.init_N(reset_N=True, N01=self.n0, N03=self.n0)
        self.p.N[2,0] = self.p.N01
        
        if dim == 1:
            self.v0 = self.p.V[1]
            # self.vn = self.p.V[-1]
        elif dim == 2:
            self.v0 = (self.p.V1[1] + self.p.V3[1]) /2
            ## We use the biggest particle for breakage in 2d
            # self.vn = (self.p.V1[-1] + self.p.V3[-1]) / 2
            # self.vn = self.p.V[-1,-1]
        # self.x0 = (6*self.v0/math.pi)**(1/3)
        # self.xn = (6*self.vn/math.pi)**(1/3)
    
    def init_mcpbe(self, dim, t, process):
        ## The number of times to repeat the MC-PBE
        self.N_MC = 5
        self.p_mc = MCPBESolver(dim=dim, verbose=True, load_attr=False, init=False)
        self.p_mc.a0 = 400
        self.p_mc.CDF_method = "disc"
        self.p.G = 1.0
        self.p_mc.process_type = process
        self.p_mc.alpha_prim = np.ones(dim**2)
        
        N = self.p.N / self.p.V_unit
        self.p_mc.n0 = np.sum(N[..., 0])
        self.p_mc.Vc = self.p_mc.a0 / self.p_mc.n0
        a_array = np.round(N[..., 0] * self.p_mc.Vc).astype(int)
        self.p_mc.V = np.zeros((dim+1, np.sum(a_array)))
        
        cnt = 0
        if dim == 1:
            for i in range(1, len(self.p.V)):
                self.p_mc.V[0, cnt:cnt + a_array[i]] = np.full(a_array[i], self.p.V[i]) 
                cnt += a_array[i]
        elif dim == 2:
            for i in range(self.p.V.shape[0]):
                for j in range(self.p.V.shape[1]):
                    if a_array[i, j] > 0:
                        self.p.V[0, cnt:cnt + a_array[i, j]] = np.full(a_array[i, j], self.p.V[i, 0])
                        self.p.V[1, cnt:cnt + a_array[i, j]] = np.full(a_array[i, j], self.p.V[0, j])
                        cnt += a_array[i, j]
        self.p_mc.V[-1, :] = np.sum(self.p_mc.V[:dim, :], axis=0)
        # self.p_mc.USE_PSD = self.use_psd
        # if process == "agglomeration" or process == "mix":
        #     self.p_mc.c = np.full(dim, self.n0*self.v0)   
        #     self.p_mc.x = np.full(dim, self.x0)
        # elif process == "breakage":
        #     self.p_mc.c = np.full(dim, self.n0*self.vn)  
        #     self.p_mc.x = np.full(dim, self.xn)
        # self.p_mc.PGV = np.full(dim, "mono")
        # self.p_mc.x2 = np.full(dim,self.x)
        # if t is not None:
        #     self.p_mc.tA = t[-1]
        #     self.p_mc.savesteps = len(t)
            
    def init_pbm(self, dim, t, process, mom_n_order, mom_n_add):
        if dim == 1:
            self.p_mom = PBMSolver(dim, t_vec=t, load_attr=False)
            self.p_mom.n_order = mom_n_order                          # Number of the simple nodes [-]
            self.p_mom.n_add = mom_n_add                          # Number of additional nodes [-] 
            self.p_mom.GQMOM = False
            self.p_mom.GQMOM_method = "lognormal"
            self.p_mom.USE_PSD = self.use_psd
            self.p_mom.process_type = process
            self.p_mom.G = 1.0
            self.p_mom.alpha_prim = np.ones(dim**2)
            self.p_mom.V_unit = 1
            self.p_mom.DIST1 = self.dist_path
            self.p_mom.DIST3 = self.dist_path
            # x_range = (0.0, self.p.V[-1])
            # if process == "agglomeration" or process == "mix":
            #     size = self.p.V[1]
            # elif process == "breakage":
            #     size = self.p.V[-1]
            ## 
            # x_range = (0.0, self.p.V[-1])
            # self.p_mom.init_moments(NDF_shape="mono",N0=self.n0, x_range=x_range, V0=self.p.V01)
            ## To better match the initial conditions of dPBE, manually initialize pbm.
            self.p_mom.x_max = self.p.V[-1]
            self.p_mom.moments = np.zeros((self.p_mom.n_order*2,self.p_mom.t_num))
            self.p_mom.moments[:,0] = np.array([np.sum(self.p.V**k*self.p.N[:,0]) for k in range(2*self.p_mom.n_order)])
            self.p_mom.normalize_mom()
            self.p_mom.set_tol()
        
    def init_mu(self):
        if self.p.t_vec is None:
            if self.kernel == "const":
                self.p.t_vec = np.arange(0, 5, 0.25, dtype=float)
            elif self.kernel == "sum":
                self.p.t_vec = np.arange(0, 100, 10, dtype=float)
        t = self.p.t_vec
        self.p_mc.tA = t[-1]
        self.p_mc.savesteps = len(t)
        
        self.mu_as = np.zeros((3,3,len(t)))
        self.mu_pbe = np.zeros((3,3,len(t)))
        self.mu_mc = np.zeros((3,3,len(t)))  
        self.mu_pbm = np.zeros((3,3,len(t)))  
        ## std_mu_mc is used for Monte-Carlo-PBESolver
        self.std_mu_mc = np.zeros((3,3,len(t)))
    
    def set_kernel_params(self, solver):
        if self.kernel == "const":
            solver.COLEVAL = 3                          
            solver.EFFEVAL = 2  
            solver.SIZEEVAL = 1
            solver.CORR_BETA = self.beta0
            
            solver.BREAKRVAL = 1
            solver.BREAKFVAL = 2
        elif self.kernel == "sum":
            solver.COLEVAL = 4                          
            solver.EFFEVAL = 2  
            solver.SIZEEVAL = 1
            solver.CORR_BETA = self.beta0 / self.v0
            solver.BREAKRVAL = 2
            solver.BREAKFVAL = 2
            
    def calculate_pbe(self):
        self.p.calc_F_M()
        self.p.calc_B_R()
        self.p.calc_int_B_F()
        self.p.solve_PBE()
        self.mu_pbe = self.p.calc_mom_t()
        
        
    def calculate_mc_pbe(self):
        mu_tmp = []
        mc_save = []
        self.p_mc.init_calc(init_Vc=False)
        for i in range(self.N_MC):
            p_mc_tem = copy.deepcopy(self.p_mc)
            p_mc_tem.solve_MC()
            mu_tmp.append(p_mc_tem.calc_mom_t())
            mc_save.append(p_mc_tem)
        self.mu_mc = np.mean(mu_tmp, axis=0)
        if self.N_MC > 1: self.std_mu_mc = np.std(mu_tmp,ddof=1,axis=0)
    
    def calculate_pbm(self):
        self.p_mom.solve_PBM()
        self.mu_pbm[:,0,:] = self.p_mom.moments[:3,:]
    
    def calculate_as_pbe(self, t=None):
        t = self.p.t_vec if t is None else t
        
        if self.kernel == "const":
            if self.p.dim == 1:
                # self.v0 = self.p.V[1]
                if self.p.process_type == "agglomeration":
                    self.mu_as[0,0,:] = 2*self.n0/(2+self.beta0*self.n0*t)
                    self.mu_as[1,0,:] = np.ones(t.shape)*self.c 
                elif self.p.process_type == "breakage":
                    print("not yet coded")
                else:
                    print("not yet coded")
            elif self.p.dim == 2:
                v10 = self.p.V1[1]
                v30 = self.p.V3[1]
                if self.p.process_type == "agglomeration":
                    n0_tot = 2*self.n0
                    self.mu_as[0,0,:] = 2*n0_tot/(2+self.beta0*n0_tot*t)
                    self.mu_as[1,0,:] = np.ones(t.shape)*self.c         
                    self.mu_as[0,1,:] = np.ones(t.shape)*self.c
                    self.mu_as[1,1,:] = self.c**2*self.beta0*n0_tot*t/n0_tot
                    self.mu_as[2,0,:] = self.c*(v10+self.c*self.beta0*n0_tot*t/n0_tot) 
                    self.mu_as[0,2,:] = self.c*(v30+self.c*self.beta0*n0_tot*t/n0_tot) 
                elif self.p.process_type == "breakage":
                    for k in range(2):
                        for l in range(2):
                            self.mu_as[k,l,:] = (self.p.V1[-1])**k*(self.p.V3[-1])**l*np.exp((2/((k+1)*(l+1))-1)*t)
                else:
                    print("Analytical solution for breakage case in 1-d not yet coded!")
                    
        elif self.kernel == "sum":        
            if self.p.dim == 1:
                if self.p.process_type == "agglomeration":
                    ### ANALYTICAL SOLUTION FROM KUMAR DISSERTATION A.11
                    self.mu_as[0,0,:] = self.n0*np.exp(-self.beta0*self.n0*t) # without self.v0, therfore p.CORR_BETA also divided by self.v0
                    self.mu_as[1,0,:] = np.ones(t.shape)*self.c
                    phi = 1-np.exp(-self.beta0*self.n0*t)
                    self.mu_as[2,0,:] = self.c*(self.v0+self.c*(2-phi)*phi/(self.n0*(1-phi)**2)) 
                elif self.p.process_type == "breakage":
                    if self.p.disc == 'uni':
                        V = self.p.V
                    else:
                        V = self.p.V_e
                    # see Kumar Dissertation A.1
                    NS = self.p.NS
                    N_as = np.zeros((NS,len(t)))
                    V_sum = np.zeros((NS,len(t)))
                    for i in range(NS):
                        for j in range(len(t)):
                            if i != NS-1:
                                N_as[i,j] = (-(t[j]*self.p.V[-1]+1)+t[j]*V[i+1])*np.exp(-V[i+1]*t[j])-\
                                    (-(t[j]*self.p.V[-1]+1)+t[j]*V[i])*np.exp(-V[i]*t[j])
                            else:
                                N_as[i,j] = (-(t[j]*self.p.V[-1]+1)+t[j]*self.p.V[i])*np.exp(-self.p.V[i]*t[j])-\
                                    (-(t[j]*self.p.V[-1]+1)+t[j]*V[i])*np.exp(-V[i]*t[j]) + \
                                    (np.exp(-t[j]*self.p.V[i]))
                            V_sum[i,j] = N_as[i,j] * self.p.V[i]
                    self.mu_as[0,0,:] = N_as.sum(axis=0)
                    self.mu_as[1,0,:] = np.ones(t.shape)*self.c 
                else:
                    self.mu_as[0,0,:] = np.ones(t.shape)*self.c 
                    self.mu_as[1,0,:] = np.ones(t.shape)*self.c 
            elif self.p.dim == 2:
                if self.p.process_type == "agglomeration":
                    ### ANALYTICAL SOLUTION FROM KUMAR DISSERTATION A.11
                    n0_tot = 2*self.n0
                    self.mu_as[0,0,:] = n0_tot*np.exp(-self.beta0*n0_tot*t) # without self.v0, therfore p.CORR_BETA also divided by self.v0
                    self.mu_as[1,0,:] = np.ones(t.shape)*self.c
                    self.mu_as[0,1,:] = np.ones(t.shape)*self.c
                    phi = 1-np.exp(-self.beta0*n0_tot*t)        
                    self.mu_as[1,1,:] = self.c**2*(2-phi)*phi/(n0_tot*(1-phi)**2)
                    self.mu_as[2,0,:] = self.c*(self.v0+self.c*(2-phi)*phi/(n0_tot*(1-phi)**2)) 
                elif self.p.process_type == "breakage":
                    ### See Leong-Table 1.
                    self.mu_as[0,1,:] =self. p.V1[-1]
                    self.mu_as[1,0,:] = self.p.V3[-1]
                    self.mu_as[0,0,:] = 1 + self.p.V[-1,-1]*t
                    # for k in range(2):
                    #     for l in range(2):
                    #         mu_as[k,l,:] = np.exp((2/((k+1)*(l+1))-1)*t)
                else:
                    for k in range(2):
                        for l in range(2):
                            self.mu_as[k,l,:] = np.ones(t.shape)*self.c
                        
    def calculate_case(self, calc_pbe=True, calc_mc=True, calc_pbm=True):
        self.init_mu()
        
        if calc_pbe:
            self.set_kernel_params(self.p)
            self.calculate_pbe()
        if calc_mc:
            self.set_kernel_params(self.p_mc)
            self.calculate_mc_pbe()
        if calc_pbm and self.p.dim == 1:
            self.set_kernel_params(self.p_mom)
            self.calculate_pbm()
        if not self.use_psd:
            self.calculate_as_pbe()
              
    def init_plot(self, default = False, size = 'half', extra = False, mrksize = 5):
        
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
                pt.plot_init(scl_a4=2, page_lnewdth_cm=18.486*2, figsze=[6.4*(18.486/13.858),4.8*(4/3)],lnewdth=0.8,
                         mrksze=mrksize*2,use_locale=True, fontsize=9*2, labelfontsize=9*2, tickfontsize=8*2)
        if default:
                pt.plot_init(scl_a4=2, page_lnewdth_cm=13.858, figsze=[6.4,4.8*(4/3)],lnewdth=0.8,
                         mrksze=mrksize,use_locale=True, fontsize=9, labelfontsize=9, tickfontsize=8)
    
    def plot_all_moments(self, ALPHA=0.7, REL=True):
        self.ax1, self.fig1 = self.plot_moment_t(i=0, j=0, label='(a)', rel=REL, alpha = ALPHA)
        self.ax2, self.fig2 = self.plot_moment_t(i=1, j=0, label='(b)', rel=REL, alpha = ALPHA)
        self.ax4, self.fig4 = self.plot_moment_t(i=2, j=0, label='(d)',labelpos='se', rel=REL, alpha = ALPHA)
        if self.p.dim == 2:
            self.ax3, self.fig3 = self.plot_moment_t(self.mu_as[:,:,1:], self.mu_pbe[:,:,1:], self.mu_mc[:,:,1:], std_mu_mc = self.std_mu_mc[:,:,1:], 
                                      i=1, j=1, t_mod=self.p.t_vec[1:], label='(c)',labelpos='se', rel=REL, alpha = ALPHA)
            
    def add_new_moments(self, NS=None, S=None, ALPHA=0.7, REL=True):
        if self.p.process_type == "breakage":
            print(
                "Breakage process does not support modifying NS or S in validation. "
                "The initial conditions of the breakage process are directly tied to NS and S. "
                "Modifying them is equivalent to changing the initial conditions.\n"
                "add_new_moments() function will be skipped."
            )
            return
        NS = self.p.NS if NS is None else NS
        S = self.p.S if S is None else S
        self.init_pbe(NS, S, self.p.dim, self.p.t_vec, self.p.disc, self.p.process_type)
        self.set_kernel_params(self.p)
        self.calculate_pbe()
        self.ax1, self.fig1 = self.add_moment_t(fig=self.fig1, ax=self.ax1, i=0, j=0, rel=REL, alpha = ALPHA)
        self.ax2, self.fig2 = self.add_moment_t(fig=self.fig2, ax=self.ax2, i=1, j=0, rel=REL, alpha = ALPHA)
        self.ax4, self.fig4 = self.add_moment_t(fig=self.fig4, ax=self.ax4, i=2, j=0, rel=REL, alpha = ALPHA)
        if self.p.dim == 2:
            self.ax3, self.fig3 = self.add_moment_t(self.mu_pbe[:,:,1:], self.fig3, self.ax3, i=1, j=1, t_mod=self.p.t_vec[1:], rel=REL, alpha = ALPHA)
          
    def plot_moment_t(self, mu_as=None, mu_pbe=None, mu_mc=None, std_mu_mc=None, t_mod=None, i=0, j=0, fig=None, ax=None, label=None,
                      labelpos='sw', rel=False, alpha=1):
        
        if fig is None or ax is None:
            fig=plt.figure()    
            ax=fig.add_subplot(1,1,1)   
        
        mu_as = self.mu_as if mu_as is None else mu_as
        mu_pbe = self.mu_pbe if mu_pbe is None else mu_pbe
        mu_mc = self.mu_mc if mu_mc is None else mu_mc
        std_mu_mc = self.std_mu_mc if std_mu_mc is None else std_mu_mc
        tp = self.p.t_vec if t_mod is None else t_mod
            
        if rel:
            ylbl = 'Relative Moment $\mu_{' + f'{i}{j}' + '}\,/\,\mu_{' + f'{i}{j}' + '}(0)$ / $-$'
        else:
            ylbl = 'Moment $\mu_{' + f'{i}{j}' + '}$ / '+'$m^{3\cdot'+str(i+j)+'}$'
            
        if mu_as is not None:
            if rel: mu_as[i,j,:] = mu_as[i,j,:]/(mu_as[i,j,0] + MIN)
            ax, fig = pt.plot_data(tp,mu_as[i,j,:], fig=fig, ax=ax,
                                   xlbl='Agglomeration time $t_\mathrm{A}$ / $s$',
                                   ylbl=ylbl, alpha=alpha,
                                   lbl='Analytical Solution',clr='k',mrk='o')

        if mu_mc is not None:
            if rel: 
                if std_mu_mc is not None:
                    std_mu_mc[i,j,:] = std_mu_mc[i,j,:]/(mu_mc[i,j,0] + MIN)
                mu_mc[i,j,:] = mu_mc[i,j,:]/(mu_mc[i,j,0] + MIN)
            
            if std_mu_mc is not None:
                ax, fig = pt.plot_data(tp,mu_mc[i,j,:], err=std_mu_mc[i,j,:], fig=fig, ax=ax,
                                       xlbl='Agglomeration time $t_\mathrm{A}$ / $s$',
                                       ylbl=ylbl, lbl='MC, $N_{\mathrm{MC}}='+str(self.N_MC)+'$',
                                       clr=c_KIT_red,mrk='s', alpha=alpha, mrkedgecolor='k')
            else:
                ax, fig = pt.plot_data(tp,mu_mc[i,j,:], fig=fig, ax=ax,
                                       xlbl='Agglomeration time $t_\mathrm{A}$ / $s$',
                                       ylbl=ylbl, lbl='MC, $N_{\mathrm{MC}}='+str(self.N_MC)+'$',
                                       clr=c_KIT_red,mrk='s', alpha=alpha, mrkedgecolor='k')
            
        if mu_pbe is not None:
            if rel: mu_pbe[i,j,:] = mu_pbe[i,j,:]/mu_pbe[i,j,0]
            ax, fig = pt.plot_data(tp,mu_pbe[i,j,:], fig=fig, ax=ax,
                                   xlbl='Agglomeration time $t_\mathrm{A}$ / $s$',
                                   ylbl=ylbl, lbl='dPBE, $N_{\mathrm{S}}='+str(self.p.NS)+'$',
                                   clr=c_KIT_green,mrk='^', alpha=alpha, mrkedgecolor='k')
        
        ## add moments from QMOM
        if self.p.dim == 1:
            mu_pbm = self.mu_pbm
            if rel:
                mu_pbm[i,j,:] = mu_pbm[i,j,:]/(mu_pbm[i,j,0] + MIN)
            ax, fig = pt.plot_data(tp,mu_pbm[i,j,:], fig=fig, ax=ax,
                                   xlbl='Agglomeration time $t_\mathrm{A}$ / $s$',
                                   ylbl=ylbl, lbl='PBM, $M_{\mathrm{order}}='+str(self.mom_order)+'$',
                                   clr=c_KIT_blue,mrk='D', alpha=alpha, mrkedgecolor='k')
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
        # plt.show()
        
        return ax, fig
    
    def add_moment_t(self, mu=None, fig=None, ax=None, i=0, j=0, lbl=None, t_mod=None, rel=False, alpha=1):
        if fig is None or ax is None:
            raise ValueError("Both 'fig' and 'ax' must be provided. Please supply them using 'plot_all_moments' or 'plot_moment_t'.")
        mu = self.mu_pbe if mu is None else mu
        tp = self.p.t_vec if t_mod is None else t_mod
            
        if lbl is None:
            lbl = 'dPBE, $N_{\mathrm{S}}='+str(self.p.NS)+'$'
        
        if rel: mu[i,j,:] = mu[i,j,:]/mu[i,j,0]
        ax, fig = pt.plot_data(tp,mu[i,j,:], fig=fig, ax=ax, alpha=alpha,
                               lbl=lbl,clr=c_KIT_green,mrk='v', mrkedgecolor='k')
        # plt.show()
        return ax, fig
    
    def show_plot(self):
        if self.fig1 is not None and self.ax1 is not None:
            plt.figure(dpi=300)
            plt.show()
        else:
            raise ValueError("No plot has been initialized to display.")