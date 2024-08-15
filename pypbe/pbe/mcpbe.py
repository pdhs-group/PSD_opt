# -*- coding: utf-8 -*-
"""
Solving multi-component population balance equations for agglomerating systems via MONTE CARLO method.
@author: Frank Rhein, frank.rhein@kit.edu, Luzia Gramling, Institute of Mechanical Process Engineering and Mechanics
"""
### ------ IMPORTS ------ ###
## General
import numpy as np
import math
import sys
from numba import jit
from scipy.stats import norm, weibull_min
import time
import copy

## For plots
import matplotlib.pyplot as plt
from ..utils.plotter import plotter as pt          
from ..utils.plotter.KIT_cmap import c_KIT_green, c_KIT_red, c_KIT_blue

### ------ POPULATION CLASS DEFINITION ------ ###
class population_MC():
    
    def __init__(self, dim=2, verbose=False):
        
        ## System parameters        
        self.c = np.full(dim,0.1e-2)              # Concentration array of components 
        self.x = np.full(dim,1e-6)                # (Mean) equivalent diameter of primary particles for each component
        self.x2 = np.full(dim,1e-6)               # (Mean) equivalent diameter of primary particles for each component (bi-modal case)
                 
        self.G = 1                                # Mean shear rate
        self.tA = 500                             # Agglomeration time [s]
        self.a0 = 1e3                             # Total amount of particles in control volume (initially) [-]
        self.savesteps = 11                       # Numer of equally spaced, saved timesteps [-]
        
        ## Initial conditions
        # PGV defines which initial particle size distribution is assumed for each component
        # 'mono': Monodisperse at x = x[i]
        # 'norm': Normal distribution at x_mean = x[i] with sigma defined in SIG 
        self.PGV = np.full(dim, 'mono') 
        self.SIG = np.full(dim,0.1)               # (relative) STD of normal distribution STD = SIG*v
        
        # PGV2 and SIG2 can be used to initialize bi-modal distributions. Use None to stay mono-modal
        self.PGV2 = None
        self.SIG2 = None
        
        ## Calculation of beta
        # BETACALC = 1: -- Random selection, beta = beta0
        # BETACALC = 2: -- Size selection, orthokinetic beta
        # BETACALC = 3: -- Size selection, beta from sum kernel 
        self.BETACALC = 1
        self.beta0 = 2.3e-18                   

        ## Calculation of alpha 
        # ALPHACALC = 1: -- Constant alpha0
        # ALPHACALC = 2: -- Calculation of alpha via collision case model
        self.ALPHACALC = 1
        self.alpha0 = 1.0
        self.alpha_prim = np.ones((dim,dim))
        
        ## Size correction of alpha
        # SIZEEVAL = 1: -- No correction
        # SIZEEVAL = 2: -- Selomulya model
        self.SIZEEVAL = 1
        self.X_SEL = 0.1                 # Size dependency parameter for Selomulya2003 / Soos2006 
        self.Y_SEL = 0.1                 # Size dependency parameter for Selomulya2003 / Soos2006
        
        ## Print more information if VERBOSE is True
        self.VERBOSE = verbose
        
        self.init_calc()
    
    def init_calc(self):
        
        # Check dimension of all relevant parameters:
        if not self.ceck_dim_consistency():
            print('Provided inputs are not consistend in their dimensions. Exiting..')
            sys.exit()
            
        # Extract number of components from concentration array c
        self.dim = len(self.c)
        
        self.v = self.x**3*math.pi/6            # Array of (mean) volume primary particles (each component)
        self.v2 = self.x2**3*math.pi/6          # Array of (mean) volume primary particles (bi-modal case)
        
        # The following part changes whether mono-or bi-modal is used
        ## Mono-modal
        if self.PGV2 is None:
            self.n = np.round(self.c/(self.v))     # Array of number concentration (each component)
            self.n2 = 0*self.n
        ## Bi-modal
        else:
            self.n = np.round(self.c/(2*self.v))   # Array of number concentration (each component)
            self.n2 = np.round(self.c/(2*self.v2)) # Array of number concentration (each component)
        
        self.n0 = np.sum([self.n,self.n2])     # Total number of primary particles
        self.Vc = self.a0/self.n0              # Control volume so that a0 is realized with respect to n0
        self.a = np.round(self.n*self.Vc).astype(int)   # Array total number of primary particles
        self.a2 = np.round(self.n2*self.Vc).astype(int) # Array total number of primary particles

        ## Calculate volume matrix
        # Each COLUMN = one individual particle / agglomerate
        # LINE no. i = partial volume component i
        # Final LINE = total volume of particle / agglomerate    
        
        # Initialize V 
        # self.V = np.zeros((self.dim+1,np.sum(self.a)))
        self.V = np.zeros((self.dim+1,np.sum([self.a,self.a2])))
        cnt = 0
        
        # Loop through all components
        for i in range (self.dim):
                
            ## Monodisperse 
            if self.PGV[i] == 'mono':
                self.V[i,cnt:cnt+self.a[i]] = np.full(self.a[i],self.v[i])
            ## Normal Distribution
            elif self.PGV[i] == 'norm':
                self.V[i,cnt:cnt+self.a[i]] = norm.rvs(loc=self.v[i], 
                                                       scale=self.SIG[i]*self.v[i], size=self.a[i])
            ## Weibull Distribution
            elif self.PGV[i] == 'weibull':
                self.V[i,cnt:cnt+self.a[i]] = weibull_min.rvs(2, loc=self.SIG[i]*self.v[i],
                                                              scale=self.v[i], size=self.a[i])
            else:
                print(f'Provided PGV "{self.PGV[i]}" is invalid')
                
            cnt += self.a[i]
            
            ## Bi-Modal
            if self.PGV2 is not None:
                
                if self.PGV2[i] == 'mono':
                    self.V[i,cnt:cnt+self.a2[i]] = np.full(self.a2[i],self.v2[i])
                ## Normal Distribution
                elif self.PGV2[i] == 'norm':
                    self.V[i,cnt:cnt+self.a2[i]] = norm.rvs(loc=self.v2[i], 
                                                           scale=self.SIG2[i]*self.v2[i], 
                                                           size=self.a2[i])
                ## Weibull Distribution
                elif self.PGV2[i] == 'weibull':
                    self.V[i,cnt:cnt+self.a2[i]] = weibull_min.rvs(2, loc=self.SIG2[i]*self.v2[i],
                                                                  scale=self.v2[i], size=self.a2[i])
                else:
                    print(f'Provided PGV "{self.PGV[i]}" is invalid')
                
                cnt += self.a2[i]
        
        # Last row: total volume
        self.V[-1,:] = np.sum(self.V[:self.dim,:], axis=0)
        
        # Delete empty entries (may occur to round-off error)
        self.V = np.delete(self.V, self.V[-1,:]==0, axis=1)
    
        self.a_tot = len(self.V[-1,:])               # Final total number of primary particles in control volume

        # IDX contains indices of initially present (primary) particles
        # Can be used to retrace which agglomerates contain which particles
        #self.IDX = [np.array([i],dtype=object) for i in range(len(self.V[-1,:]))]
        self.IDX = [[i] for i in range(len(self.V[-1,:]))]

        #self.IDX = np.array(self.IDX,dtype=object)
        
        # Calculate equivalent diameter from total volume 
        self.X=(6*(self.V[-1,:])/math.pi)**(1/3)
              
        # Initialize time array
        self.t=[0]
        
        # Initialize beta array
        if self.BETACALC == 2 or self.BETACALC == 3:
            self.betaarray = calc_betaarray_jit(self.BETACALC, self.a_tot, self.G, self.X, self.beta0, self.V)
        
        # Save arrays
        self.t_save = np.linspace(0,self.tA,self.savesteps)
        self.V_save = [self.V]
        self.Vc_save = [self.Vc]
        self.V0 = self.V
        self.X0 = self.X
        self.V0_save = [self.V0]
        self.IDX_save = [copy.deepcopy(self.IDX)]

        self.step=1    
    
    # Solve MC N times
    def solve_MC_N(self, N=5, maxiter=1e8):
        
        # Copy base class instance (save all parameters)
        base = copy.deepcopy(self)
        
        if self.VERBOSE:
            cnt = 1
            print(f'### Calculating MC iteration no. {cnt}/{N}.. ###')
            
        # Solve for the first time 
        self.solve_MC(maxiter)
        
        # Loop through all iterations (-1, first calculation already done)
        for n in range(N-1):
            if self.VERBOSE:
                cnt += 1
                print(f'### Calculating MC iteration no. {cnt}/{N}.. ###')
                
            # Set up temporary class instance, initialize it again (randomness in PSD) and solve it
            temp = copy.deepcopy(base)
            temp.init_calc()
            temp.solve_MC(maxiter)
            
            # Integrate temporary instance in main class instance
            self.combine_MC(temp)
    
    # Solve MC 1 time
    def solve_MC(self, maxiter=1e8):
        
        # Iteration counter and initial time
        count = 0
        t0 = time.time()
        
        while self.t[-1] <= self.tA and count < maxiter:
        
            ## Simplified case for random choice of collision partners (constant kernel) 
            if self.BETACALC == 1:
                
                self.beta = self.beta0
                idx1, idx2 = self.select_random()
                self.betaarray=0                
            
            ## Calculation of beta array containing all collisions
            ## Probability-weighted choice of two collision partners
            elif self.BETACALC == 2 or self.BETACALC == 3:
                
                # select = self.select_size()                    # Class method
                select = select_size_jit(self.betaarray[0,:])    # JIT-compiled select
                self.beta = self.betaarray[0,select]
                idx1 = int(self.betaarray[1,select])
                idx2 = int(self.betaarray[2,select])
                
            ## Calculation of alpha
            if self.ALPHACALC == 1:            
                self.alpha = self.alpha0 
            elif self.ALPHACALC == 2:
                self.alpha = self.calc_alpha_ccm(idx1,idx2)
            
            ## Size-correction
            if self.SIZEEVAL == 2:
                lam = min([self.X[idx1]/self.X[idx2],self.X[idx2]/self.X[idx1]])
                alpha_corr = np.exp(-self.X_SEL*(1-lam)**2) / ((self.V[-1,idx1]*self.V[-1,idx2]/(np.mean(self.V0[-1,:])**2))**self.Y_SEL)
                self.alpha *= alpha_corr

            # Check if agglomeration occurs (if random number e[0,1] is smaller than alpha)
            if self.alpha > np.random.rand():
                ## Modification of V, X and IDX at idx1. 
                self.V[:,idx1] = self.V[:,idx1] + self.V[:,idx2]
                self.X[idx1] = (6*(self.V[self.dim,idx1])/math.pi)**(1/3)  
                
                #self.IDX[idx1] = np.append(self.IDX[idx1],self.IDX[idx2])
                self.IDX[idx1] += self.IDX[idx2]
                
                ## Deletion of agglomerate at idx 2
                self.V = np.delete(self.V, idx2, axis=1)   
                self.X = np.delete(self.X, idx2)  
                #self.IDX = np.delete(self.IDX, idx2) 
                del self.IDX[idx2]
                self.a_tot -= 1 
                
                ## New calculation of beta array
                self.betaarray = calc_betaarray_jit(self.BETACALC, self.a_tot, self.G, self.X, self.beta0, self.V)
            
            ## Calculation of inter event time
            dt = self.calc_inter_event_time()                                    # Class method
            #dt = calc_inter_event_time_array(self.Vc,self.a_tot,self.betaarray)  # JIT-compiled
            
            ## Add current timestep               
            self.t=np.append(self.t,self.t[-1]+dt)

            ## When total amount of particles is less than half of its original value 
            ## --> duplicate all agglomerates and double control volume
            if self.a_tot <= self.a0/2:
                self.Vc = self.Vc*2
                self.a_tot = self.a_tot*2
                self.V = np.append(self.V, self.V, axis=1)
                self.X = np.append(self.X, self.X)  
                # Double indices, Add total number of initially present particles
                # No value appears double and references to [V_save[0], V_save[0], ...]
                tmp = copy.deepcopy(self.IDX)
                for i in range(len(tmp)):
                    for j in range(len(tmp[i])):
                        tmp[i][j]+=len(self.V0[0,:])
                self.IDX = self.IDX + tmp
                # self.IDX = np.append(self.IDX, self.IDX + len(self.V0[0,:]))
                self.V0 = np.append(self.V0, self.V0, axis=1)
                
                if self.VERBOSE:
                    print(f'## Doubled control volume {int(np.log2(self.Vc*self.n0/self.a0))}-time(s). Current time:  {int(self.t[-1])}s/{self.tA}s ##')
            
            ## Save at specific times
            if self.t_save[self.step] <= self.t[-1]:
                self.t_save[self.step] = self.t[-1]
                self.V_save.append(self.V)
                self.Vc_save.append(self.Vc)
                self.V0_save.append(self.V0)                
                self.IDX_save = self.IDX_save + [copy.deepcopy(self.IDX)]
                self.step+=1
                
            count += 1
        
        # Calculation (machine) time
        self.MACHINE_TIME = time.time()-t0
        
        if self.VERBOSE:
            # Print why calculations stopped
            if count == maxiter:
                print('XX Maximum number of iterations reached XX')
                print(f'XX Final calculation time is {int(self.t[-1])}s/{self.tA}s XX')
            else:
                print(f'## Agglomeration time reached after {count} iterations ##')
            print(f'## The calculation took {int(self.MACHINE_TIME)}s ##')
        
    # Random choice of two agglomeration partners. Return indices.
    def select_random(self):
        idx1 = np.random.randint(self.a_tot)
        idx2 = np.random.randint(self.a_tot)
        while idx1 == idx2:
            idx2 = np.random.randint(self.a_tot)
        return idx1, idx2    
    
    # Beta-weighted selection (requires beta array)
    def select_size(self): 
        return np.random.choice(np.arange(self.betaarray.shape[1]),
                                p=self.betaarray[0,:]/np.sum(self.betaarray[0,:]))       
    
    # Calculation of inter-event-time          
    def calc_inter_event_time(self):
        #Berechnung der inter-event-time nach Briesen (2008) mit konstantem/berechneten/ausgewähtem beta
        if self.BETACALC == 1 :
            dtd=2*self.Vc/(self.beta0*self.a_tot**2)
        
        elif self.BETACALC == 2 or self.BETACALC == 3: 
            # Use mean value of all betas
            dtd=2*self.Vc/(self.a_tot**2*np.mean(self.betaarray[0,:]))
          
        return dtd   
    
    # Calculate alpha base on collision case model
    def calc_alpha_ccm(self, idx1, idx2):
        P = np.zeros((self.dim,self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                P[i,j] = (self.V[i,idx1]/self.V[-1,idx1])*(self.V[j,idx2]/self.V[-1,idx2])
        return np.sum(P*self.alpha_prim)
        
    # Calculate distribution moments mu(i,j,t)
    def calc_mom_t(self):
        
        mu = np.zeros((3,3,len(self.t_save)))
        
        # Time loop
        for t in range(len(self.t_save)):
            
            for i in range(3):
                if self.dim == 1:
                    mu[i,0,t] = np.sum(self.V_save[t][0,:]**i/self.Vc_save[t])
                    
                # The following only applies for more than 1 component
                else:
                    for j in range(3):
                        mu[i,j,t] = np.sum(self.V_save[t][0,:]**i*self.V_save[t][1,:]**j/self.Vc_save[t])
        
        self.mu = mu
        
        return mu
    
    ## Combine multiple class instances (if t-grid is identical / for repeated calculations)
    def combine_MC(self,m):
        # Check dimensions
        if len(self.Vc_save) != len(m.Vc_save):
            print("Cannot combine two instances as t arrays don't align. Discarding iteration..")
            return
            
        # Combine data for each timestep
        for t in range(len(self.t_save)):
            self.t_save[t] = np.mean([self.t_save[t],m.t_save[t]])
            
            # Control volumina add up
            #print(len(self.Vc_save[t]),len(m.Vc_save[t]))
            self.Vc_save[t] += m.Vc_save[t]
            
            # Generating new idices and appending them (IDX of m needs to be raised)
            tmp = copy.deepcopy(m.IDX_save[t])
            for i in range(len(tmp)):
                for j in range(len(tmp[i])):
                    tmp[i][j]+=len(self.V0_save[t][0,:])
            
            self.IDX_save[t] = self.IDX_save[t] + tmp
            
            # Combining initial state V0 
            self.V0_save[t] = np.append(self.V0_save[t], m.V0_save[t], axis=1)
            self.V_save[t] = np.append(self.V_save[t], m.V_save[t], axis=1)
                            
    ## Visualize / plot density and sum distribution:
    def visualize_qQ_t(self,t_plot=None,t_pause=0.5,close_all=False,scl_a4=1,figsze=[12.8,6.4*1.5],
                       show_x10=False, show_x50=True, show_x90=False):
        # Definition of t_plot:
        # None: Plot all available times
        # Numpy array in range [0,1] --> Relative values of time indices
        # E.g. t_plot=np.array([0,0.5,1]) plots start, half-time and end
        
        import seaborn as sns
        import pandas as pd
        
        # Initialize plot
        pt.plot_init(scl_a4=scl_a4,figsze=figsze,lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
            
        if close_all:
            plt.close('all')
        
        fig=plt.figure()    
        ax1=fig.add_subplot(1,2,1) 
        ax2=fig.add_subplot(1,2,2)
        
        if t_plot is None:
            t_plot = np.arange(len(self.t_save))
        else:
            t_plot = np.round(t_plot*(len(self.t_save)-1)).astype(int)
        
        xmin = min(self.return_distribution(t=t_plot[0])[0])*1e6
        xmax = max(self.return_distribution(t=t_plot[-1])[0])*1e6
        
        for t in t_plot:
            
            # Calculate distribution
            x_uni, q3, Q3, x_10, x_50, x_90 = self.return_distribution(t=t)
            
            ax1.cla()
            ax2.cla()
            
            ax1, fig = pt.plot_data(x_uni*1e6, q3, ax=ax1, fig=fig, plt_type='scatter',
                                xlbl='Particle Diameter $x$ / $\mu$m',
                                ylbl='Volume density distribution $q_3$ / $-$',
                                clr='k',mrk='o',leg=False,zorder=2)
            
            if len(x_uni) > 1:
                sns.histplot(data=pd.DataFrame({'x':x_uni[q3>=0]*1e6,'q':q3[q3>=0]}), x='x',weights='q', 
                              log_scale=True, bins=15, ax=ax1, kde=True, color=c_KIT_green)
            
            ax2, fig = pt.plot_data(x_uni*1e6, Q3, ax=ax2, fig=fig,
                                xlbl='Particle Diameter $x$ / $\mu$m',
                                ylbl='Volume sum distribution $Q_3$ / $-$',
                                clr='k',mrk='o',leg=False)
            
            #ax1.set_ylim([0,0.25])
            ax1.grid('minor')
            ax2.grid('minor')
            ax1.set_xscale('log')
            ax2.set_xscale('log')
            ax1.set_xlim([xmin,xmax])
            ax2.set_xlim([xmin,xmax])
            ax1.set_ylim([0,1])
            ax2.set_ylim([0,1])
            
            if show_x10: ax1.axvline(x_10*1e6, color=c_KIT_green)
            if show_x10: ax2.axvline(x_10*1e6, color=c_KIT_green)
            if show_x50: ax1.axvline(x_50*1e6, color=c_KIT_red)
            if show_x50: ax2.axvline(x_50*1e6, color=c_KIT_red)
            if show_x90: ax1.axvline(x_90*1e6, color=c_KIT_blue)
            if show_x90: ax2.axvline(x_90*1e6, color=c_KIT_blue)
            plt.tight_layout() 
            plt.pause(t_pause)
    
        plt.show()        
        
        return ax1, ax2, fig
    
    ## Visualize Moments
    def visualize_mom_t(self, i=0, j=0, fig=None, ax=None, clr=c_KIT_green, lbl='MC'):
        
        if fig is None or ax is None:
            fig=plt.figure()    
            ax=fig.add_subplot(1,1,1)   
        
        ax, fig = pt.plot_data(self.t_save,self.mu[i,j,:], fig=fig, ax=ax,
                               xlbl='Agglomeration time $t_\mathrm{A}$ / $s$',
                               ylbl=f'Moment $\mu ({i}{j})$ / '+'$m^{3\cdot'+str(i+j)+'}$',
                               lbl=lbl,clr=clr,mrk='o')
                
        # Adjust y scale in case of first moment
        if i+j == 1:
            ax.set_ylim([np.min(self.mu[i,j,:])*0.9,np.max(self.mu[i,j,:])*1.1])
        ax.grid('minor')
        plt.tight_layout()   
        
        return ax, fig 
    
    ## Return particle size distribution 
    def return_distribution(self, comp='all', t=0, Q3_grid=None):
        
        v_uni = np.array([])
        sumvol_uni = np.array([])
        
        if comp == 'all':
            # Extract unique values of V and save corresponding volume if not empty
            for i in range(len(self.V_save[t][-1,:])):
                if not self.V_save[t][-1,i] in v_uni:
                    v_uni = np.append(v_uni,self.V_save[t][-1,i])
                    sumvol_uni = np.append(sumvol_uni,self.V_save[t][-1,i]) 
                    
                else:
                    sumvol_uni[v_uni == self.V_save[t][-1,i]] += self.V_save[t][-1,i] 

            # Sort v_uni in ascending order and keep track in sumvol_uni
            v_uni = v_uni[np.argsort(v_uni)]
            sumvol_uni = sumvol_uni[np.argsort(v_uni)]
            
            # Calculate diameter array
            x_uni=(6*v_uni/np.pi)**(1/3)
            
            # Calculate sum and density distribution
            Q3 = np.zeros(len(v_uni))
            Q3[1:] = np.cumsum(sumvol_uni[1:])/np.sum(sumvol_uni[1:])
            q3 = sumvol_uni/np.sum(sumvol_uni)
            
            # Retrieve x10, x50 and x90 through interpolation
            x_10=np.interp(0.1, Q3, x_uni)
            x_50=np.interp(0.5, Q3, x_uni)
            x_90=np.interp(0.9, Q3, x_uni)
            
        else:
            print('Case for comp not coded yet. Exiting')
            return
        
        # Q3_grid: 1D np.array to define extrapolation steps
        if Q3_grid is not None:
            x_uni_grid = np.interp(Q3_grid, Q3, x_uni)
            q3_grid = np.interp(x_uni_grid, x_uni, q3)
            
            return x_uni_grid, q3_grid, Q3_grid, x_10, x_50, x_90
        
        else:
            return x_uni, q3, Q3, x_10, x_50, x_90
    
    def ceck_dim_consistency(self):
        check = np.array([len(self.x),len(self.PGV),len(self.SIG),
                          self.alpha_prim.shape[0],self.alpha_prim.shape[1]])
        return len(check[check==len(self.c)]) == len(check)

## JIT-compiled calculation of beta array 
@jit(nopython=True)
def calc_betaarray_jit(BETACALC, a, G, X, beta0, V):            
    ## There are a*(a-1) possible collisions, however all are counted twice (e.g [1,0]=[0,1])
    num = int(a*(a-1)/2)
    ## Beta array is defined as follows: 
    # Columns: Collision pairs
    # Line 0: Beta value
    # Line 1: Index partner 1
    # Line 2: Index partner 2
    beta = np.zeros((3, num))
    cnt = 0
    
    # Orthokinetic kernel
    if BETACALC == 2:
        for i in range(a):
            for j in range(i):
                beta[0,cnt] = beta0*G*2.3*((3*V[-1,i]/(4*np.pi))**(1/3)+(3*V[-1,j]/(4*np.pi))**(1/3))**3
                beta[1,cnt] = i
                beta[2,cnt] = j
                cnt += 1  
                
    # Sum-kernel
    if BETACALC == 3:
        for i in range(a):
            for j in range(i):
                beta[0,cnt] = beta0*(V[-1,i]+V[-1,j])
                beta[1,cnt] = i
                beta[2,cnt] = j
                cnt += 1        
                
    return beta

@jit(nopython=True)
def calc_alpha_ccm_jit(V, alpha_prim, idx1, idx2):
    dim = len(V[:,0])
    P = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            P[i,j] = (V[i,idx1]/V[-1,idx1])*(V[j,idx2]/V[-1,idx2])
    return np.sum(P*alpha_prim)

## JIT-compiled calculation of inter-event-time   
@jit(nopython=True)       
def calc_inter_event_time_array(Vc,a_tot,betaarray):
    return 2*Vc/(a_tot**2*np.mean(betaarray[0,:]))

@jit(nopython=True)
def select_size_jit(betaarray):
    return np.arange(len(betaarray))[np.searchsorted(np.cumsum(betaarray/np.sum(betaarray)),
                                                     np.random.random(), side="right")]
    
            
    