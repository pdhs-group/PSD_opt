# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 08:53:21 2025

@author: px2030
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from optframework.pbe import DPBESolver, ExtruderPBESolver
import optframework.utils.plotter.plotter as pt
from optframework.utils.plotter.KIT_cmap import c_KIT_green, c_KIT_red, c_KIT_blue

class PBEValidation():
    def __init__(self, dim, grid, NS, S, kernel, process,
                 c=1, x=2e-6, beta0=1e-16, t=None, NC=3, Extruder=False):
        self.x = x
        self.beta0 = beta0
        self.kernel = kernel
        self.c = c
        self.n0 = 3*c/(4*math.pi*(x/2)**3)
        if not Extruder:
            self.p = DPBESolver(dim=dim, t_vec=t, load_attr=False,f=grid)
        else:
            self.p = ExtruderPBESolver(dim=dim, NC=NC, t_vec=t, load_attr=False,disc=grid)
        self.init_pbe_params(NS, S, dim, kernel, process)
        self.choose_case()
        
        ## parameters for Monte-Carlo-PBESolver
        ## MC-Solver not yet coded
        self.N_MC = 5
    def init_pbe(self, NS, S, dim, kernel, process):
        self.p.NS = NS
        self.p.S = S
        self.p.R01, self.p.R03 = self.x/2, self.x/2
        self.p.USE_PSD = False
        self.p.alpha_prim = np.ones(dim**2)
        self.p.process_type = process
        self.p.calc_R()
        
    def init_mu(self, t):
        self.mu_as = np.zeros((3,3,len(t)))
        self.mu_pbe = np.zeros((3,3,len(t)))
        self.mu_mc = np.zeros((3,3,len(t)))  
        ## std_mu_mc is used for Monte-Carlo-PBESolver
        self.std_mu_mc = np.zeros((3,3,len(t)))
        
    def calculate_pbe(self):
        self.p.init_N(reset_N=True, N01=self.n0, N03=self.n0)
        self.p.calc_F_M()
        self.p.calc_B_R()
        self.p.calc_int_B_F()
        self.p.solve_PBE()
        self.mu_pbe = self.p.calc_mom_t()
        
    def calculate_case(self):
        if self.kernel == "const":
            if self.p.t_vec is None:
                self.p.t_vec = np.arange(0, 5, 0.25, dtype=float)
            self.init_mu(self.p.t_vec)
            
            self.p.COLEVAL = 3                          
            self.p.EFFEVAL = 2  
            self.p.SIZEEVAL = 1
            self.p.CORR_BETA = self.beta0
            
            self.p.BREAKFVAL = 2
            self.p.BREAKRVAL = 1
            self.calculate_pbe()
            
            t = self.p.t_vec
            if self.p.dim == 1:
                # v0 = self.p.V[1]
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
                    print("not yet coded")
                    
        elif self.kernel == "sum":
            if self.p.t_vec is None:
                self.p.t_vec = np.arange(0, 100, 10, dtype=float)
            self.init_mu(self.p.t_vec)
            self.p.COLEVAL = 4                          
            self.p.EFFEVAL = 2  
            self.p.SIZEEVAL = 1
            if self.p.dim == 1:
               v0 = self.p.V[1]
            elif self.p.dim == 2:
               v10 = self.p.V1[1]
               v30 = self.p.V3[1]
               v0 = (v10 + v30) /2
            self.p.CORR_BETA = self.beta0 / v0
            self.p.BREAKFVAL = 2
            self.p.BREAKRVAL = 2
            self.calculate_pbe()
            
            t = self.p.t_vec
            if self.p.dim == 1:
                if self.p.process_type == "agglomeration":
                    ### ANALYTICAL SOLUTION FROM KUMAR DISSERTATION A.11
                    self.mu_as[0,0,:] = self.n0*np.exp(-self.beta0*self.n0*t) # without v0, therfore p.CORR_BETA also divided by v0
                    self.mu_as[1,0,:] = np.ones(t.shape)*self.c
                    phi = 1-np.exp(-self.beta0*self.n0*t)
                    self.mu_as[2,0,:] = self.c*(v0+self.c*(2-phi)*phi/(self.n0*(1-phi)**2)) 
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
                    self.mu_as[0,0,:] = n0_tot*np.exp(-self.beta0*n0_tot*t) # without v0, therfore p.CORR_BETA also divided by v0
                    self.mu_as[1,0,:] = np.ones(t.shape)*self.c
                    self.mu_as[0,1,:] = np.ones(t.shape)*self.c
                    phi = 1-np.exp(-self.beta0*n0_tot*t)        
                    self.mu_as[1,1,:] = self.c**2*(2-phi)*phi/(n0_tot*(1-phi)**2)
                    self.mu_as[2,0,:] = self.c*(v0+self.c*(2-phi)*phi/(n0_tot*(1-phi)**2)) 
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
    
    def plot_all_moments(self, ALPHA=0.7, REL=True):
        self.ax1, self.fig1 = self.plot_moment_t(i=0, j=0, label='(a)', rel=REL, alpha = ALPHA)
        self.ax2, self.fig2 = self.plot_moment_t(i=1, j=0, label='(b)', rel=REL, alpha = ALPHA)
        self.ax4, self.fig4 = self.plot_moment_t(i=2, j=0, label='(d)',labelpos='se', rel=REL, alpha = ALPHA)
        if self.p.dim == 2:
            self.ax3, self.fig3 = self.plot_moment_t(self.mu_as[:,:,1:], self.mu_pbe[:,:,1:], self.mu_mc[:,:,1:], std_mu_mc = self.std_mu_mc[:,:,1:], 
                                      i=1, j=1, t_mod=self.p.t_vec[1:], label='(c)',labelpos='se', rel=REL, alpha = ALPHA)
            
    def add_new_moments(self, NS=None, S=None, ALPHA=0.7, REL=True):
        NS = self.p.NS if NS is None else NS
        S = self.p.S if S is None else S
        self.init_pbe(self, NS, S, self.p.dim, self.kernel, self.p.process_type)
        self.calculate_pbe()
        self.ax1, self.fig1 = self.add_moment_t(self.fig1, self.ax1, i=0, j=0, rel=REL, alpha = ALPHA)
        self.ax2, self.fig2 = self.add_moment_t(self.fig2, self.ax2, i=1, j=0, rel=REL, alpha = ALPHA)
        self.ax4, self.fig4 = self.add_moment_t(self.fig4, self.ax4, i=2, j=0, rel=REL, alpha = ALPHA)
        if self.p.dim == 2:
            self.ax3, self.fig3 = self.add_moment_t(self.mu_pbe2[:,:,1:], self.fig3, self.ax3, i=1, j=1, t_mod=self.p.t_vec[1:], rel=REL, alpha = ALPHA)
          
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
    
    def add_moment_t(self, mu, fig, ax, i=0, j=0, lbl=None, t_mod=None, rel=False, alpha=1):
        mu = self.mu_pbe if mu is None else mu
        tp = self.p.t_vec if t_mod is None else t_mod
            
        if lbl is None:
            lbl = 'dPBE, $N_{\mathrm{S}}='+str(self.p.NS)+'$'
        
        if rel: mu[i,j,:] = mu[i,j,:]/mu[i,j,0]
        ax, fig = pt.plot_data(tp,mu[i,j,:], fig=fig, ax=ax, alpha=alpha,
                               lbl=lbl,clr=c_KIT_green,mrk='v', mrkedgecolor='k')
        
        return ax, fig