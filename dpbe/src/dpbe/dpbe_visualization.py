# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 09:45:12 2024

@author: Administrator

Enhanced DPBE Visualization Module
Provides comprehensive visualization capabilities for discrete Population Balance Equation (PBE) solvers.
Supports both 1D and 2D particle systems with moment evolution, distribution analysis, and animations.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from pbe_core.func.static_method import KDE_fit, KDE_score
from pbe_core.plotter import plotter as pt
from pbe_core.plotter.KIT_cmap import c_KIT_green, c_KIT_red, c_KIT_blue
        
class DPBEVisual:
    def __init__(self, base):
        self.base = base
        
    def init_visual_params(self):
        """Initialize visualization parameters and settings."""
        pass

    # ========================================
    # MOMENT AND TEMPORAL EVOLUTION METHODS
    # ========================================
    
    def visualize_moments_vs_time(self, moment_indices=[(0,0), (1,0), (2,0)], 
                                 normalize=True, ax=None, fig=None, 
                                 close_all=False, lbl='', clr='k', mrk='o',
                                 scl_a4=1, figsze=[12.8, 6.4*1.5]):
        """
        Visualize specific moments vs time.
        
        Parameters
        ----------
        moment_indices : list of tuples
            List of (i,j) moment indices to plot. For 1D: only i matters. For 2D: both i,j matter.
        normalize : bool
            Whether to normalize by initial values.
        """
        base = self.base
        pt.plot_init(scl_a4=scl_a4, figsze=figsze, lnewdth=0.8, mrksze=5, use_locale=True, scl=1.2)
        
        if close_all:
            plt.close('all')
            
        if fig is None or ax is None:
            fig = plt.figure()    
            ax = fig.add_subplot(1, 1, 1)
        
        # Calculate moments
        mu = base.post.calc_mom_t()
        
        for idx, (i, j) in enumerate(moment_indices):
            if base.dim == 1:
                moment_data = mu[i, 0, :]  # For 1D, only first index matters
                label = f'μ_{i,0}' if not lbl else f'{lbl}_μ_{i,0}'
            else:  # 2D case
                moment_data = mu[i, j, :]
                label = f'μ_{i,j}' if not lbl else f'{lbl}_μ_{i,j}'
            
            if normalize and moment_data[0] != 0:
                moment_data = moment_data / moment_data[0]
                ylabel = 'Normalized moments μ(i,j,t)/μ(i,j,0) [-]'
            else:
                ylabel = 'Moments μ(i,j,t) [-]'
            
            colors = [clr, c_KIT_red, c_KIT_blue, c_KIT_green] if clr == 'k' else [clr]
            color = colors[idx % len(colors)]
            
            ax, fig = pt.plot_data(base.t_vec, moment_data, fig=fig, ax=ax,
                                  xlbl='Time t [s]',
                                  ylbl=ylabel,
                                  lbl=label, clr=color, mrk=mrk)
        
        ax.grid('minor')
        ax.legend()
        plt.tight_layout()
        
        return ax, fig

    def visualize_total_number_vs_time(self, normalize=True, ax=None, fig=None,
                                      close_all=False, lbl='', clr='k', mrk='o',
                                      scl_a4=1, figsze=[12.8, 6.4*1.5]):
        """
        Visualize total particle number vs time.
        This is equivalent to the (0,0) moment.
        """
        return self.visualize_moments_vs_time(
            moment_indices=[(0,0)], normalize=normalize, ax=ax, fig=fig,
            close_all=close_all, lbl=lbl if lbl else 'Total Number', 
            clr=clr, mrk=mrk, scl_a4=scl_a4, figsze=figsze
        )

    def visualize_total_volume_vs_time(self, normalize=True, ax=None, fig=None,
                                      close_all=False, lbl='', clr='k', mrk='o',
                                      scl_a4=1, figsze=[12.8, 6.4*1.5]):
        """
        Visualize total particle volume vs time.
        This is equivalent to the (1,0) moment for 1D or (1,0)+(0,1) for 2D.
        """
        if self.base.dim == 1:
            moment_indices = [(1,0)]
            lbl_default = 'Total Volume (1D)'
        else:
            moment_indices = [(1,0), (0,1)]
            lbl_default = 'Total Volume (2D)'
            
        return self.visualize_moments_vs_time(
            moment_indices=moment_indices, normalize=normalize, ax=ax, fig=fig,
            close_all=close_all, lbl=lbl if lbl else lbl_default,
            clr=clr, mrk=mrk, scl_a4=scl_a4, figsze=figsze
        )

    def visualize_average_volume_vs_time(self, normalize=True, ax=None, fig=None,
                                        close_all=False, lbl='', clr='k', mrk='o',
                                        scl_a4=1, figsze=[12.8, 6.4*1.5]):
        """
        Visualize average particle volume vs time.
        Average volume = Total volume / Total number.
        """
        base = self.base
        pt.plot_init(scl_a4=scl_a4, figsze=figsze, lnewdth=0.8, mrksze=5, use_locale=True, scl=1.2)
        
        if close_all:
            plt.close('all')
            
        if fig is None or ax is None:
            fig = plt.figure()    
            ax = fig.add_subplot(1, 1, 1)
        
        # Calculate moments
        mu = base.post.calc_mom_t()
        
        # Calculate average volume
        if base.dim == 1:
            total_number = mu[0, 0, :]
            total_volume = mu[1, 0, :]
        else:  # 2D case
            total_number = mu[0, 0, :]
            total_volume = mu[1, 0, :] + mu[0, 1, :]
        
        # Avoid division by zero
        avg_volume = np.divide(total_volume, total_number, 
                              out=np.zeros_like(total_volume), 
                              where=total_number!=0)
        
        if normalize and avg_volume[0] != 0:
            avg_volume = avg_volume / avg_volume[0]
            ylabel = 'Normalized average volume V̄/V̄₀ [-]'
        else:
            ylabel = 'Average volume V̄ [m³]'
        
        label = lbl if lbl else 'Average Volume'
        
        ax, fig = pt.plot_data(base.t_vec, avg_volume, fig=fig, ax=ax,
                              xlbl='Time t [s]',
                              ylbl=ylabel,
                              lbl=label, clr=clr, mrk=mrk)
        
        ax.grid('minor')
        plt.tight_layout()
        
        return ax, fig

    def visualize_percentiles_vs_time(self, percentiles=[10, 50, 90], normalize=True,
                                     ax=None, fig=None, close_all=False, lbl='',
                                     scl_a4=1, figsze=[12.8, 6.4*1.5]):
        """
        Visualize x_10, x_50, x_90 vs time.
        
        Parameters
        ----------
        percentiles : list
            Which percentiles to plot (e.g., [10, 50, 90]).
        normalize : bool
            Whether to normalize by initial values.
        """
        base = self.base
        pt.plot_init(scl_a4=scl_a4, figsze=figsze, lnewdth=0.8, mrksze=5, use_locale=True, scl=1.2)
        
        if close_all:
            plt.close('all')
            
        if fig is None or ax is None:
            fig = plt.figure()    
            ax = fig.add_subplot(1, 1, 1)
        
        colors = [c_KIT_red, c_KIT_blue, c_KIT_green]
        markers = ['o', 's', '^']
        
        # Calculate percentiles for each time step
        percentile_data = {p: [] for p in percentiles}
        
        for t in range(len(base.t_vec)):
            results = base.post.return_distribution(t=t, flag='x_10,x_50,x_90')
            x_10, x_50, x_90 = results
            
            percentile_values = {'10': x_10, '50': x_50, '90': x_90}
            
            for p in percentiles:
                percentile_data[p].append(percentile_values[str(p)])
        
        # Plot each percentile
        for idx, p in enumerate(percentiles):
            data = np.array(percentile_data[p])
            
            if normalize and data[0] != 0:
                data = data / data[0]
                ylabel = f'Normalized x_p/x_p₀ [-]'
            else:
                ylabel = 'Particle diameter x_p [μm]'
            
            label = f'{lbl}_x_{p}' if lbl else f'x_{p}'
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            
            ax, fig = pt.plot_data(base.t_vec, data, fig=fig, ax=ax,
                                  xlbl='Time t [s]',
                                  ylbl=ylabel,
                                  lbl=label, clr=color, mrk=marker)
        
        ax.grid('minor')
        ax.legend()
        plt.tight_layout()
        
        return ax, fig

    # ========================================
    # DISTRIBUTION VISUALIZATION METHODS
    # ========================================
    
    def visualize_distribution_at_time(self, t=-1, dist_type='q3', smoothing=False,
                                      ax=None, fig=None, close_all=False,
                                      log_x_axis=True, lbl='', clr='k', mrk='o',
                                      scl_a4=1, figsze=[12.8, 6.4*1.5]):
        """
        Visualize PSD (q, Q, or Weibull) at a specific time.
        
        Parameters
        ----------
        t : int
            Time index (-1 for final time).
        dist_type : str
            Type of distribution: 'q3', 'Q3', 'weibull', or 'all'.
        """
        base = self.base
        pt.plot_init(scl_a4=scl_a4, figsze=figsze, lnewdth=0.8, mrksze=5, use_locale=True, scl=1.2)
        
        if close_all:
            plt.close('all')
        
        # Get distribution data
        results = base.post.return_distribution(t=t, flag='all')
        x_uni, qx, Qx, x_10, x_50, x_90, sum_uni, x_weibull, y_weibull = results
        
        if smoothing and dist_type in ['q3', 'all']:
            if len(x_uni) > 1 and np.sum(sum_uni[1:]) > 0:
                kde = KDE_fit(x_uni[1:], sum_uni[1:])
                qx_smooth = KDE_fit(kde, x_uni[1:])
                qx = np.insert(qx_smooth, 0, 0.0)
        
        if dist_type == 'all':
            if fig is None or ax is None:
                fig = plt.figure(figsize=[figsze[0]*1.5, figsze[1]])
                ax1 = fig.add_subplot(1, 3, 1)
                ax2 = fig.add_subplot(1, 3, 2)
                ax3 = fig.add_subplot(1, 3, 3)
                axes = [ax1, ax2, ax3]
            else:
                axes = ax if isinstance(ax, list) else [ax]
            
            # q3 distribution
            axes[0], fig = pt.plot_data(x_uni, qx, fig=fig, ax=axes[0],
                                       xlbl='Particle diameter x [μm]',
                                       ylbl='Volume density q₃ [-]',
                                       lbl=lbl, clr=clr, mrk=mrk)
            
            # Q3 distribution  
            axes[1], fig = pt.plot_data(x_uni, Qx, fig=fig, ax=axes[1],
                                       xlbl='Particle diameter x [μm]',
                                       ylbl='Cumulative volume Q₃ [-]',
                                       lbl=lbl, clr=clr, mrk=mrk)
            
            # Weibull distribution
            if len(x_weibull) > 0:
                axes[2], fig = pt.plot_data(x_weibull, y_weibull, fig=fig, ax=axes[2],
                                           xlbl='ln(x) [μm]',
                                           ylbl='ln(ln(1/(1-Q₃)))',
                                           lbl=lbl, clr=clr, mrk=mrk)
            
            for ax_i in axes:
                ax_i.grid('minor')
                if log_x_axis and ax_i != axes[2]:  # Don't log-scale Weibull plot
                    ax_i.set_xscale('log')
            
            return axes, fig
            
        else:
            if fig is None or ax is None:
                fig = plt.figure()    
                ax = fig.add_subplot(1, 1, 1)
            
            if dist_type == 'q3':
                ax, fig = pt.plot_data(x_uni, qx, fig=fig, ax=ax,
                                      xlbl='Particle diameter x [μm]',
                                      ylbl='Volume density q₃ [-]',
                                      lbl=lbl, clr=clr, mrk=mrk)
            elif dist_type == 'Q3':
                ax, fig = pt.plot_data(x_uni, Qx, fig=fig, ax=ax,
                                      xlbl='Particle diameter x [μm]',
                                      ylbl='Cumulative volume Q₃ [-]',
                                      lbl=lbl, clr=clr, mrk=mrk)
            elif dist_type == 'weibull':
                if len(x_weibull) > 0:
                    ax, fig = pt.plot_data(x_weibull, y_weibull, fig=fig, ax=ax,
                                          xlbl='ln(x) [μm]',
                                          ylbl='ln(ln(1/(1-Q₃)))',
                                          lbl=lbl, clr=clr, mrk=mrk)
                    log_x_axis = False  # Weibull is already log-transformed
            
            ax.grid('minor')
            if log_x_axis:
                ax.set_xscale('log')
            
            plt.tight_layout()
            return ax, fig

    def visualize_distribution_animation(self, t_vec=None, dist_type='q3', smoothing=False,
                                        fps=5, log_x_axis=True, save_path=None,
                                        figsze=[12.8, 6.4*1.5]):
        """
        Create animation of PSD evolution over time.
        
        Parameters
        ----------
        t_vec : array-like, optional
            Time vector. If None, uses base.t_vec.
        dist_type : str
            Type of distribution: 'q3', 'Q3', or 'weibull'.
        fps : int
            Frames per second for animation.
        save_path : str, optional
            Path to save animation (e.g., 'animation.gif').
        """
        base = self.base
        
        if t_vec is None:
            t_vec = base.t_vec
            
        fig = plt.figure(figsize=figsze)
        ax = fig.add_subplot(1, 1, 1)
        
        # Get initial data for axis limits
        results_init = base.post.return_distribution(t=0, flag='all')
        results_final = base.post.return_distribution(t=-1, flag='all')
        
        x_min = min(results_init[0][results_init[0] > 0])
        x_max = max(results_final[0])
        
        def update(frame):
            ax.clear()
            
            results = base.post.return_distribution(t=frame, flag='all')
            x_uni, qx, Qx, x_10, x_50, x_90, sum_uni, x_weibull, y_weibull = results
            
            if smoothing and dist_type == 'q3':
                if len(x_uni) > 1 and np.sum(sum_uni[1:]) > 0:
                    kde = KDE_fit(x_uni[1:], sum_uni[1:])
                    qx = KDE_fit(kde, x_uni[1:])
                    qx = np.insert(qx, 0, 0.0)
            
            label = f't = {t_vec[frame]:.2f} s'
            
            if dist_type == 'q3':
                ax.plot(x_uni, qx, label=label, color='b', marker='o', markersize=3)
                ax.set_ylabel('Volume density q₃ [-]')
            elif dist_type == 'Q3':
                ax.plot(x_uni, Qx, label=label, color='r', marker='s', markersize=3)
                ax.set_ylabel('Cumulative volume Q₃ [-]')
            elif dist_type == 'weibull':
                if len(x_weibull) > 0:
                    ax.plot(x_weibull, y_weibull, label=label, color='g', marker='^', markersize=3)
                    ax.set_ylabel('ln(ln(1/(1-Q₃)))')
                    ax.set_xlabel('ln(x) [μm]')
                    log_x_axis = False
            
            if dist_type != 'weibull':
                ax.set_xlabel('Particle diameter x [μm]')
                
            ax.grid('minor')
            ax.legend()
            
            if log_x_axis:
                ax.set_xscale('log')
                ax.set_xlim([x_min, x_max])
        
        t_frames = np.arange(len(t_vec))
        
        ani = FuncAnimation(fig, update, frames=t_frames, blit=False, 
                           interval=1000/fps, repeat=True)
        
        if save_path:
            ani.save(save_path, writer='pillow' if save_path.endswith('.gif') else 'ffmpeg')
        
        plt.tight_layout()
        return ani

    # ========================================
    # DISCRETE DISTRIBUTION VISUALIZATION
    # ========================================
    
    def visualize_discrete_distribution(self, t=-1, ax=None, fig=None, close_all=False,
                                       log_scale=True, show_colorbar=True,
                                       scl_a4=1, figsze=[12.8, 6.4*1.5]):
        """
        Visualize discrete particle number distribution N.
        - 1D: Line plot of N vs particle class
        - 2D: Heatmap/contour plot of N on the grid
        """
        base = self.base
        pt.plot_init(scl_a4=scl_a4, figsze=figsze, lnewdth=0.8, mrksze=5, use_locale=True, scl=1.2)
        
        if close_all:
            plt.close('all')
        
        # Get particle distribution at time t
        if hasattr(base, 'N') and base.N is not None:
            if t == -1:
                N_t = base.N[:, -1] if base.dim == 1 else base.N[:, :, -1]
            else:
                N_t = base.N[:, t] if base.dim == 1 else base.N[:, :, t]
        else:
            print("No particle distribution data available.")
            return None, None
        
        if base.dim == 1:
            # 1D case: Line plot
            if fig is None or ax is None:
                fig = plt.figure()    
                ax = fig.add_subplot(1, 1, 1)
            
            # Remove boundary values
            N_plot = N_t[1:-1]  # Remove first and last elements (boundary)
            indices = np.arange(1, len(N_t)-1)
            
            # Only plot non-zero values
            nonzero_mask = N_plot > 0
            if np.any(nonzero_mask):
                ax.plot(indices[nonzero_mask], N_plot[nonzero_mask], 
                       'o-', color='b', linewidth=2, markersize=4)
                
                if log_scale:
                    ax.set_yscale('log')
                
                ax.set_xlabel('Particle class index')
                ax.set_ylabel('Number concentration N [#/m³]')
                ax.set_title(f'1D Discrete Distribution at t = {base.t_vec[t]:.2f} s')
                ax.grid(True, alpha=0.3)
        
        else:
            # 2D case: Heatmap
            if fig is None or ax is None:
                fig = plt.figure()    
                ax = fig.add_subplot(1, 1, 1)
            
            # Remove boundary values
            N_plot = N_t[1:-1, 1:-1]
            
            # Create coordinate arrays
            x_coords = np.arange(N_plot.shape[1])
            y_coords = np.arange(N_plot.shape[0])
            X, Y = np.meshgrid(x_coords, y_coords)
            
            # Use log scale if requested and data allows it
            if log_scale and np.any(N_plot > 0):
                N_plot_log = np.log10(N_plot + 1e-20)  # Add small value to avoid log(0)
                N_plot_log[N_plot <= 0] = np.nan  # Set zero/negative values to NaN
                
                im = ax.contourf(X, Y, N_plot_log, levels=20, cmap='viridis')
                if show_colorbar:
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('log₁₀(N) [log(#/m³)]')
            else:
                im = ax.contourf(X, Y, N_plot, levels=20, cmap='viridis')
                if show_colorbar:
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('N [#/m³]')
            
            ax.set_xlabel('Component 1 class index')
            ax.set_ylabel('Component 2 class index')
            ax.set_title(f'2D Discrete Distribution at t = {base.t_vec[t]:.2f} s')
        
        plt.tight_layout()
        return ax, fig

    # ========================================
    # LEGACY METHODS (MAINTAINED FOR COMPATIBILITY)
    # ========================================
    def visualize_distN_t(self,t_plot=None,t_pause=0.5,close_all=False,scl_a4=1,figsze=[12.8,6.4*1.5]):
        base = self.base
        # Definition of t_plot:
        # None: Plot all available times
        # Numpy array in range [0,1] --> Relative values of time indices
        # E.g. t_plot=np.array([0,0.5,1]) plots start, half-time and end
        
        # Double figsize in 3-D case
        if base.dim == 1 or base.dim == 2: 
            pt.plot_init(scl_a4=scl_a4,figsze=figsze,lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
        else: 
            pt.plot_init(scl_a4=4,frac_lnewdth=2,figsze=figsze,lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
            
        if close_all:
            plt.close('all')
        
        fig=plt.figure()    
        
        if t_plot is None:
            tmp = None
            t_plot = np.arange(len(base.t_vec))
        else:
            t_plot = np.round(t_plot*(len(base.t_vec)-1))
            
        # 1-D case: Plot PSD over time        
        if base.dim == 1:
            print('For 1-D case executing visualize_qQ_t instead.')
            ax1, ax2, fig = self.visualize_qQ_t(t_plot=tmp,t_pause=t_pause,close_all=close_all,
                                                scl_a4=scl_a4,figsze=figsze,
                                                show_x10=False, show_x50=True, show_x90=False)
            return [ax1, ax2], fig
        
        # 2-D case: Plot distribution over time
        elif base.dim == 2:
            ax=fig.add_subplot(1,1,1)
            
            for t in t_plot:
                
                if 'cb' in locals(): cb.remove()                
                ax,cb,fig = self.plot_N2D(base.N[1:,1:,t],base.V[1:,1:],np.sum(base.N[1:,1:,0]*base.V[1:,1:]),
                                          ax=ax,fig=fig,t_stamp=f'{np.round(base.t_vec[t])}s')
            
                plt.pause(t_pause)
        
            plt.show()
            return ax, fig
            
        # 3-D case: Plot distributions over time   
        elif base.dim ==3:
            ax1 = fig.add_subplot(2,2,1)
            ax2 = fig.add_subplot(2,2,2)
            ax3 = fig.add_subplot(2,2,3)
            ax4 = fig.add_subplot(2,2,4)
            
            #Calculate date for distribution plot:
            Xt=np.zeros((3,len(t_plot)))
    
            for t in t_plot:
                Ntmp=base.N[:,:,:,t]
                Nagg=np.sum(Ntmp)-np.sum(Ntmp[:,1,1])-np.sum(Ntmp[1,:,1])-np.sum(Ntmp[1,1,:])
                if not Nagg == 0:
                    Xt[0,t]=np.sum(Ntmp[base.X1_vol!=1]*base.X1_vol[base.X1_vol!=1])/Nagg
                    Xt[1,t]=np.sum(Ntmp[base.X2_vol!=1]*base.X2_vol[base.X2_vol!=1])/Nagg
                    Xt[2,t]=np.sum(Ntmp[base.X3_vol!=1]*base.X3_vol[base.X3_vol!=1])/Nagg
                else:
                    Xt[0,t] = Xt[1,t] = Xt[2,t] = 0 
            
            Xt[:,0]=Xt[:,1]
            
            for t in t_plot:
                
                if 'cb1' in locals(): cb1.remove()
                if 'cb2' in locals(): cb2.remove()    
                if 'cb3' in locals(): cb3.remove()
                ax1,cb1,fig = self.plot_N2D(base.N[1:,1:,1,t],base.V[1:,1:,1],np.sum(base.N[:,:,:,0]*base.V),
                                            ax=ax1,fig=fig)
                ax2,cb2,fig = self.plot_N2D(base.N[1:,1,1:,t],base.V[1:,1,1:],np.sum(base.N[:,:,:,0]*base.V),
                                            ax=ax2,fig=fig)
                ax3,cb3,fig = self.plot_N2D(base.N[1,1:,1:,t],base.V[1,1:,1:],np.sum(base.N[:,:,:,0]*base.V),
                                            ax=ax3,fig=fig)
                
                
                ax2.set_xlabel('Partial volume comp. 3 $V_{3}$ ($k$) / $-$')  # Add a y-label to the axes.
                ax3.set_ylabel('Partial volume comp. 2 $V_{2}$ ($k$) / $-$')  # Add a y-label to the axes.
                ax3.set_xlabel('Partial volume comp. 3 $V_{3}$ ($k$) / $-$')  # Add a y-label to the axes.
                
                ax4.cla()
                ax4, fig = pt.plot_data(base.t_vec[:t+1],Xt[0,:t+1], fig=fig, ax=ax4,
                                        xlbl='Agglomeration time $t_\mathrm{A}$ / $-$',
                                        ylbl='Agglomerate composition / $-$',
                                        lbl=None,clr='k',plt_type='line',leg=False)
                ax4, fig = pt.plot_data(base.t_vec[:t+1],Xt[1,:t+1]+Xt[0,:t+1], 
                                        fig=fig, ax=ax4, lbl=None,clr='k',plt_type='line',leg=False)
                ax4, fig = pt.plot_data(base.t_vec[:t+1],Xt[2,:t+1]+Xt[1,:t+1]+Xt[0,:t+1], 
                                        fig=fig, ax=ax4, lbl=None,clr='k',plt_type='line',leg=False)
                ax4.stackplot(base.t_vec[:t+1],Xt[:,:t+1],colors=[c_KIT_green,c_KIT_red,c_KIT_blue],
                              labels=['Comp. 1','Comp. 2','Comp. 3'])
                ax4.legend(loc='upper right')
                ax4.text(0.05, 0.95, f'{np.round(base.t_vec[t])}s', transform=ax4.transAxes, fontsize=10*1.6,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='w', alpha=1))
                
                ax4.set_xlim([0,base.t_vec[-1]])
                ax4.set_ylim([0,1])
                plt.tight_layout()
                plt.pause(t_pause)
        
            plt.show()
        
            return [ax1, ax2, ax3, ax4], fig
    
    def visualize_qQ_t(self,t_plot=None,t_pause=0.5,close_all=False,scl_a4=1,figsze=[12.8,6.4*1.5],
                       show_x10=False, show_x50=True, show_x90=False):
        base = self.base
        # Definition of t_plot:
        # None: Plot all available times
        # Numpy array in range [0,1] --> Relative values of time indices
        # E.g. t_plot=np.array([0,0.5,1]) plots start, half-time and end
        
        # Initialize plot
        pt.plot_init(scl_a4=scl_a4,figsze=figsze,lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
            
        if close_all:
            plt.close('all')
        
        fig=plt.figure()    
        ax1=fig.add_subplot(1,2,1) 
        ax2=fig.add_subplot(1,2,2)
        
        if t_plot is None:
            t_plot = np.arange(len(base.t_vec))
        else:
            t_plot = np.round(t_plot*(len(base.t_vec)-1)).astype(int)
        
        # Get initial and final distributions for axis limits
        results_init = base.post.return_distribution(t=t_plot[0], flag='all')
        results_final = base.post.return_distribution(t=t_plot[-1], flag='all')
        
        xmin = min(results_init[0][results_init[0] > 0]) if np.any(results_init[0] > 0) else 1e-6
        xmax = max(results_final[0])
        
        for t in t_plot:
    
            # Calculate distribution
            results = base.post.return_distribution(t=t, flag='all')
            x_uni, q3, Q3, x_10, x_50, x_90 = results[:6]
            
            ax1.cla()
            ax2.cla()
            
            ax1, fig = pt.plot_data(x_uni, q3, ax=ax1, fig=fig, plt_type='scatter',
                                xlbl='Particle Diameter x [μm]',
                                ylbl='Volume density distribution q₃ [-]',
                                clr='k',mrk='o',leg=False,zorder=2)
            
            ax2, fig = pt.plot_data(x_uni, Q3, ax=ax2, fig=fig,
                                xlbl='Particle Diameter x [μm]',
                                ylbl='Volume sum distribution Q₃ [-]',
                                clr='k',mrk='o',leg=False)
            
            ax1.grid('minor')
            ax2.grid('minor')
            ax1.set_xscale('log')
            ax2.set_xscale('log')
            ax1.set_xlim([xmin, xmax])
            ax2.set_xlim([xmin, xmax])
            
            if show_x10: 
                ax1.axvline(x_10, color=c_KIT_green, alpha=0.7, label='x₁₀')
                ax2.axvline(x_10, color=c_KIT_green, alpha=0.7, label='x₁₀')
            if show_x50: 
                ax1.axvline(x_50, color=c_KIT_red, alpha=0.7, label='x₅₀')
                ax2.axvline(x_50, color=c_KIT_red, alpha=0.7, label='x₅₀')
            if show_x90: 
                ax1.axvline(x_90, color=c_KIT_blue, alpha=0.7, label='x₉₀')
                ax2.axvline(x_90, color=c_KIT_blue, alpha=0.7, label='x₉₀')
                
            plt.tight_layout() 
            plt.pause(t_pause)
    
        plt.show()        
        
        return ax1, ax2, fig
    
    ## Visualize / plot population:
    def visualize_sumN_t(self,ax=None,fig=None,close_all=False,lbl='',clr='k',mrk='o',scl_a4=1,figsze=[12.8,6.4*1.5]):
        base = self.base
        pt.plot_init(scl_a4=scl_a4,figsze=figsze,lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
        if close_all:
            plt.close('all')
            
        if fig is None or ax is None:
            fig=plt.figure()    
            ax=fig.add_subplot(1,1,1)   
        
        # Use post processing method to get total number
        total_N = base.post.return_N_t()
        
        ax, fig = pt.plot_data(base.t_vec, total_N, fig=fig, ax=ax,
                               xlbl='Time t [s]',
                               ylbl='Total number of particles Σ N [-]',
                               lbl=lbl,clr=clr,mrk=mrk)
            
        ax.grid('minor')
        plt.tight_layout()   
        
        return ax, fig
    
    def visualize_sumvol_t(self, sumvol=None, ax=None,fig=None,close_all=False,lbl='',clr='k',mrk='o',scl_a4=1,figsze=[12.8,6.4*1.5]):
        base = self.base
        pt.plot_init(scl_a4=scl_a4,figsze=figsze,lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
        if close_all:
            plt.close('all')
            
        if fig is None or ax is None:
            fig=plt.figure()    
            ax=fig.add_subplot(1,1,1)   
        
        # Calculate total volume using moments
        mu = base.post.calc_mom_t()
        if base.dim == 1:
            total_volume = mu[1, 0, :]
        else:  # 2D case
            total_volume = mu[1, 0, :] + mu[0, 1, :]
        
        # Use provided sumvol if available, otherwise use calculated
        plot_data = sumvol if sumvol is not None else total_volume
        
        ax, fig = pt.plot_data(base.t_vec, plot_data, fig=fig, ax=ax,
                               xlbl='Time t [s]',
                               ylbl='Total volume of particles Σ V [m³]',
                               lbl=lbl,clr=clr,mrk=mrk)
        
        ax.grid('minor')
        plt.tight_layout()   
        
        return ax, fig
    
    def visualize_distribution(self, q3=None, Q3=None, t=-1, smoothing=False, vol_dis=True,
                               axq3=None,axQ3=None, fig=None,close_all=False,log_x_axis=True, 
                               lbl='',clr='k',mrk='o',scl_a4=1,figsze=[12.8,6.4*1.5]): 
        if q3 is None or Q3 is None:
            if vol_dis:
                results = self.base.post.return_distribution(t=t, flag='all')
                x_uni, q3, Q3 = results[0], results[1], results[2]
                sum_uni = results[6]
                ylbl = 'Volume distribution q₃ [-]'
            else:
                results = self.base.post.return_distribution(t=t, flag='all', q_type='q0')
                x_uni, q3, Q3 = results[0], results[1], results[2]
                sum_uni = results[6]
                ylbl = 'Number distribution q₀ [-]'
        else:
            ylbl = 'Distribution [-]'
            
        if smoothing and len(x_uni) > 1:
            if np.sum(sum_uni[1:]) > 0:
                kde = KDE_fit(x_uni[1:], sum_uni[1:])
                q3 = KDE_fit(kde, x_uni[1:])
                q3 = np.insert(q3, 0, 0.0)
            
        pt.plot_init(scl_a4=scl_a4,figsze=figsze,lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
        if close_all:
            plt.close('all')
            
        if fig is None or axq3 is None or axQ3 is None:
            fig=plt.figure()    
            axq3=fig.add_subplot(1,2,1)   
            axQ3=fig.add_subplot(1,2,2)   
        
        axq3, fig = pt.plot_data(x_uni, q3, fig=fig, ax=axq3,
                               xlbl='Particle diameter x [μm]',
                               ylbl=ylbl,
                               lbl=lbl,clr=clr,mrk=mrk)
        
        axQ3, fig = pt.plot_data(x_uni, Q3, fig=fig, ax=axQ3,
                               xlbl='Particle diameter x [μm]',
                               ylbl='Cumulative distribution Q [-]',
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
                kde = KDE_fit(x_uni[1:], sum_uni[1:])
                q3 = KDE_score(kde, x_uni[1:])
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
                        kde_other = KDE_fit(x_uni_other[1:], sum_uni_other[1:])
                        q3_other = KDE_score(kde_other, x_uni_other[1:])
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
            t_vec = self.base.t_vec
        t_frame = np.arange(len(t_vec))
        axq3.set_xlabel('Agglomeration size $x_\mathrm{A}$ / $-$')
        axq3.set_ylabel(ylbl)
        axq3.grid('minor')
        if log_x_axis:
            axq3.set_xscale('log')
        plt.tight_layout()
    
        ani = FuncAnimation(fig, update, frames=t_frame, blit=False)
        return ani

    def plot_N2D(self, N, V, N_total, factor=1000, ax=None, fig=None):
        """
        Plot 2D number distribution for discrete PBE.
        
        Args:
            N (array): Number distribution array
            V (array): Volume array
            N_total (float): Total number for normalization
            factor (float): Scaling factor for better visualization
            ax: Matplotlib axis object
            fig: Matplotlib figure object
            
        Returns:
            tuple: (ax, colorbar, fig) matplotlib objects
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        from matplotlib.patches import Rectangle
        
        if fig is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        elif ax is None:
            ax = fig.add_subplot(111)
        
        # Scale the data for visualization
        N_scaled = N * factor
        
        # Create heatmap
        im = ax.imshow(N_scaled, origin='lower', aspect='auto', 
                      cmap='viridis', interpolation='nearest')
        
        # Set labels and title
        ax.set_xlabel('Particle size index j')
        ax.set_ylabel('Particle size index i') 
        ax.set_title(f'2D Number Distribution (scaled by {factor})')
        
        # Add colorbar
        cb = plt.colorbar(im, ax=ax)
        cb.set_label(f'Number density × {factor}')
        
        plt.tight_layout()
        return ax, cb, fig
    
    def return_distribution(self, t=-1, flag='all', q_type='q3'):
        """
        Wrapper method for backward compatibility.
        
        Args:
            t (int): Time index (-1 for last time point)
            flag (str): Type of data to return
            q_type (str): Distribution type ('q3' for volume, 'q0' for number)
            
        Returns:
            tuple: Distribution data based on flag parameter
        """
        # Delegate to post-processing method
        return self.base.post.return_distribution(t=t, flag=flag, q_type=q_type)

