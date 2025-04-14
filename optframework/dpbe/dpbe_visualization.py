# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 09:45:12 2024

@author: Administrator
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from optframework.kernel_opt.opt_data import KDE_fit, KDE_score
from optframework.utils.plotter import plotter as pt          
from optframework.utils.plotter.KIT_cmap import c_KIT_green, c_KIT_red, c_KIT_blue
        
def init_visual_params(self):
    pass
## Visualize / plot population:
def visualize_distN_t(self,t_plot=None,t_pause=0.5,close_all=False,scl_a4=1,figsze=[12.8,6.4*1.5]):
    # Definition of t_plot:
    # None: Plot all available times
    # Numpy array in range [0,1] --> Relative values of time indices
    # E.g. t_plot=np.array([0,0.5,1]) plots start, half-time and end
    
    # Double figsize in 3-D case
    if self.dim == 1 or self.dim == 2: 
        pt.plot_init(scl_a4=scl_a4,figsze=figsze,lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
    else: 
        pt.plot_init(scl_a4=4,frac_lnewdth=2,figsze=figsze,lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
        
    if close_all:
        plt.close('all')
    
    fig=plt.figure()    
    
    if t_plot is None:
        tmp = None
        t_plot = np.arange(len(self.t_vec))
    else:
        t_plot = np.round(t_plot*(len(self.t_vec)-1))
        
    # 1-D case: Plot PSD over time        
    if self.dim == 1:
        print('For 1-D case executing visualize_qQ_t instead.')
        ax1, ax2, fig = self.visualize_qQ_t(t_plot=tmp,t_pause=t_pause,close_all=close_all,
                                            scl_a4=scl_a4,figsze=figsze,
                                            show_x10=False, show_x50=True, show_x90=False)
        return [ax1, ax2], fig
    
    # 2-D case: Plot distribution over time
    elif self.dim == 2:
        ax=fig.add_subplot(1,1,1)
        
        for t in t_plot:
            
            if 'cb' in locals(): cb.remove()                
            ax,cb,fig = self.plot_N2D(self.N[1:,1:,t],self.V[1:,1:],np.sum(self.N[1:,1:,0]*self.V[1:,1:]),
                                      ax=ax,fig=fig,t_stamp=f'{np.round(self.t_vec[t])}s')
        
            plt.pause(t_pause)
    
        plt.show()
        return ax, fig
        
    # 3-D case: Plot distributions over time   
    elif self.dim ==3:
        ax1 = fig.add_subplot(2,2,1)
        ax2 = fig.add_subplot(2,2,2)
        ax3 = fig.add_subplot(2,2,3)
        ax4 = fig.add_subplot(2,2,4)
        
        #Calculate date for distribution plot:
        Xt=np.zeros((3,len(t_plot)))

        for t in t_plot:
            Ntmp=self.N[:,:,:,t]
            Nagg=np.sum(Ntmp)-np.sum(Ntmp[:,1,1])-np.sum(Ntmp[1,:,1])-np.sum(Ntmp[1,1,:])
            if not Nagg == 0:
                Xt[0,t]=np.sum(Ntmp[self.X1_vol!=1]*self.X1_vol[self.X1_vol!=1])/Nagg
                Xt[1,t]=np.sum(Ntmp[self.X2_vol!=1]*self.X2_vol[self.X2_vol!=1])/Nagg
                Xt[2,t]=np.sum(Ntmp[self.X3_vol!=1]*self.X3_vol[self.X3_vol!=1])/Nagg
            else:
                Xt[0,t] = Xt[1,t] = Xt[2,t] = 0 
        
        Xt[:,0]=Xt[:,1]
        
        for t in t_plot:
            
            if 'cb1' in locals(): cb1.remove()
            if 'cb2' in locals(): cb2.remove()    
            if 'cb3' in locals(): cb3.remove()
            ax1,cb1,fig = self.plot_N2D(self.N[1:,1:,1,t],self.V[1:,1:,1],np.sum(self.N[:,:,:,0]*self.V),
                                        ax=ax1,fig=fig)
            ax2,cb2,fig = self.plot_N2D(self.N[1:,1,1:,t],self.V[1:,1,1:],np.sum(self.N[:,:,:,0]*self.V),
                                        ax=ax2,fig=fig)
            ax3,cb3,fig = self.plot_N2D(self.N[1,1:,1:,t],self.V[1,1:,1:],np.sum(self.N[:,:,:,0]*self.V),
                                        ax=ax3,fig=fig)
            
            
            ax2.set_xlabel('Partial volume comp. 3 $V_{3}$ ($k$) / $-$')  # Add a y-label to the axes.
            ax3.set_ylabel('Partial volume comp. 2 $V_{2}$ ($k$) / $-$')  # Add a y-label to the axes.
            ax3.set_xlabel('Partial volume comp. 3 $V_{3}$ ($k$) / $-$')  # Add a y-label to the axes.
            
            ax4.cla()
            ax4, fig = pt.plot_data(self.t_vec[:t+1],Xt[0,:t+1], fig=fig, ax=ax4,
                                    xlbl='Agglomeration time $t_\mathrm{A}$ / $-$',
                                    ylbl='Agglomerate composition / $-$',
                                    lbl=None,clr='k',plt_type='line',leg=False)
            ax4, fig = pt.plot_data(self.t_vec[:t+1],Xt[1,:t+1]+Xt[0,:t+1], 
                                    fig=fig, ax=ax4, lbl=None,clr='k',plt_type='line',leg=False)
            ax4, fig = pt.plot_data(self.t_vec[:t+1],Xt[2,:t+1]+Xt[1,:t+1]+Xt[0,:t+1], 
                                    fig=fig, ax=ax4, lbl=None,clr='k',plt_type='line',leg=False)
            ax4.stackplot(self.t_vec[:t+1],Xt[:,:t+1],colors=[c_KIT_green,c_KIT_red,c_KIT_blue],
                          labels=['Comp. 1','Comp. 2','Comp. 3'])
            ax4.legend(loc='upper right')
            ax4.text(0.05, 0.95, f'{np.round(self.t_vec[t])}s', transform=ax4.transAxes, fontsize=10*1.6,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='w', alpha=1))
            
            ax4.set_xlim([0,self.t_vec[-1]])
            ax4.set_ylim([0,1])
            plt.tight_layout()
            plt.pause(t_pause)
    
        plt.show()
    
        return [ax1, ax2, ax3, ax4], fig

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
        t_plot = np.arange(len(self.t_vec))
    else:
        t_plot = np.round(t_plot*(len(self.t_vec)-1)).astype(int)
    
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

## Visualize / plot population:
def visualize_sumN_t(self,ax=None,fig=None,close_all=False,lbl='',clr='k',mrk='o',scl_a4=1,figsze=[12.8,6.4*1.5]):
    
    pt.plot_init(scl_a4=scl_a4,figsze=figsze,lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
    if close_all:
        plt.close('all')
        
    if fig is None or ax is None:
        fig=plt.figure()    
        ax=fig.add_subplot(1,1,1)   
    
    ax, fig = pt.plot_data(self.t_vec,self.return_N_t(), fig=fig, ax=ax,
                           xlbl='Agglomeration time $t_\mathrm{A}$ / $-$',
                           ylbl='Total number of agglomerates $\Sigma N$ / $-$',
                           lbl=lbl,clr=clr,mrk=mrk)
        
    ax.grid('minor')
    plt.tight_layout()   
    
    return ax, fig

def visualize_sumvol_t(self, sumvol, ax=None,fig=None,close_all=False,lbl='',clr='k',mrk='o',scl_a4=1,figsze=[12.8,6.4*1.5]):
    
    pt.plot_init(scl_a4=scl_a4,figsze=figsze,lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
    if close_all:
        plt.close('all')
        
    if fig is None or ax is None:
        fig=plt.figure()    
        ax=fig.add_subplot(1,1,1)   
    
    ax, fig = pt.plot_data(self.t_vec,sumvol, fig=fig, ax=ax,
                           xlbl='Agglomeration time $t_\mathrm{A}$ / $-$',
                           ylbl='Total volume of agglomerates $\Sigma N$ / $-$',
                           lbl=lbl,clr=clr,mrk=mrk)
    
    if self.dim == 1:
        mu = self.calc_mom_t()
        mu_10 = mu[1, 0, :]
        ax, fig = pt.plot_data(self.t_vec,mu_10, fig=fig, ax=ax,
                               lbl=lbl,clr=clr,mrk='^')
        
    if self.dim == 2:
        
        mu = self.calc_mom_t()
        mu_10 = mu[1, 0, :] + mu[0, 1, :]
        ax, fig = pt.plot_data(self.t_vec,mu_10, fig=fig, ax=ax,
                               lbl=lbl,clr=clr,mrk='v')
    
    ax.grid('minor')
    plt.tight_layout()   
    
    return ax, fig

def visualize_distribution(self, q3=None, Q3=None, t=-1, smoothing=False, vol_dis=True,
                           axq3=None,axQ3=None, fig=None,close_all=False,log_x_axis=True, 
                           lbl='',clr='k',mrk='o',scl_a4=1,figsze=[12.8,6.4*1.5]): 
    if q3 is None or Q3 is None:
        if vol_dis:
            x_uni, q3, Q3, sum_uni = self.return_distribution(t=t, flag='x_uni, qx, Qx, sum_uni')
            ylbl = 'volume distribution of agglomerates $q3$ / $-$'
        else:
            x_uni, q3, Q3, sum_uni = self.return_distribution(t=t, flag='x_uni, qx, Qx, sum_uni', q_type= 'q0')
            ylbl = 'number distribution of agglomerates $q3$ / $-$'
    if smoothing:
        kde = KDE_fit(None, x_uni[1:], sum_uni[1:])
        q3 = KDE_score(self, kde, x_uni[1:])
        q3 = np.insert(q3, 0, 0.0)
        
    pt.plot_init(scl_a4=scl_a4,figsze=figsze,lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
    if close_all:
        plt.close('all')
        
    if fig is None or axq3 is None or axQ3 is None:
        fig=plt.figure()    
        axq3=fig.add_subplot(1,2,1)   
        axQ3=fig.add_subplot(1,2,2)   
        
    
    axq3, fig = pt.plot_data(x_uni, q3, fig=fig, ax=axq3,
                           xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                           ylbl=ylbl,
                           lbl=lbl,clr=clr,mrk=mrk)
    
    axQ3, fig = pt.plot_data(x_uni, Q3, fig=fig, ax=axQ3,
                           xlbl='Agglomeration size $x_\mathrm{A}$ / $-$',
                           ylbl=ylbl,
                           lbl=lbl,clr=clr,mrk=mrk)

    axq3.grid('minor')
    axQ3.grid('minor')
    if log_x_axis:
        axq3.set_xscale('log')
        axQ3.set_xscale('log')
    
    plt.tight_layout()   
    
    return axq3, axQ3, fig

def visualize_distribution_animation(self, t_vec=None, smoothing=False, 
                                     vol_dis=True,axq3=None, fig=None,fps=5,
                                     log_x_axis=True,others=None, other_labels=None):
    if fig is None or axq3 is None:
        fig=plt.figure()    
        axq3=fig.add_subplot(1,1,1)    
    def update(frame):
        q3lbl = f"t={t_vec[frame]}"
        while len(axq3.lines) > 0:
            axq3.lines[0].remove()
            
        if vol_dis:
            x_uni, q3, Q3, sum_uni = self.return_distribution(t=frame, flag='x_uni, qx, Qx, sum_uni')
        else:
            x_uni, q3, Q3, sum_uni = self.return_distribution(t=frame, flag='x_uni, qx, Qx, sum_uni', q_type= 'q0')
        if smoothing:
            kde = KDE_fit(None, x_uni[1:], sum_uni[1:])
            q3 = KDE_score(self, kde, x_uni[1:])
            q3 = np.insert(q3, 0, 0.0)
        axq3.plot(x_uni, q3, label=q3lbl, color='b', marker='o')
        # 绘制其他实例的结果
        if others is not None:
            colors = ['r', 'g', 'm', 'c', 'y']
            for i, other in enumerate(others):
                if vol_dis:
                    x_uni_other, q3_other, Q3_other, sum_uni_other = other.return_distribution(t=frame, flag='x_uni, qx, Qx, sum_uni')
                else:
                    x_uni_other, q3_other, Q3_other, sumvol__other = other.return_distribution(t=frame, flag='x_uni, qx, Qx, sum_uni', q_type= 'q0') 
                if smoothing:
                    kde_other = KDE_fit(None, x_uni_other[1:], sum_uni_other[1:])
                    q3_other = KDE_score(self, kde_other, x_uni_other[1:])
                    q3_other = np.insert(q3_other, 0, 0.0)
                    
                label = other_labels[i] if other_labels and i < len(other_labels) else f"Other {i+1} (t={t_vec[frame]})"
                axq3.plot(x_uni_other, q3_other, label=label, color=colors[i % len(colors)], marker='^')
        axq3.legend()
        return axq3,

    if vol_dis:
        ylbl = 'volume distribution of agglomerates $q3$ / $-$'
    else:
        ylbl = 'number distribution of agglomerates $q3$ / $-$'
    if t_vec is None:
        t_vec = self.t_vec
    t_frame = np.arange(len(t_vec))
    axq3.set_xlabel('Agglomeration size $x_\mathrm{A}$ / $-$')
    axq3.set_ylabel(ylbl)
    axq3.grid('minor')
    if log_x_axis:
        axq3.set_xscale('log')
    plt.tight_layout()

    ani = FuncAnimation(fig, update, frames=t_frame, blit=False)
    return ani

