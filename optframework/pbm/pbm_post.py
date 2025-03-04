# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 08:58:15 2025

@author: px2030
"""
import numpy as np
import matplotlib.pyplot as plt
import optframework.utils.plotter.plotter as pt

class PBMPost:
    def __init__(self, solver):
        self.solver = solver

    def plot_moments_comparison(self, moments, moments_QMOM, moments_GQMOM):
        """
        Plot a visual comparison of QMOM and GQMOM moments against original moments.

        Parameters:
            moments (array-like): Original moments (true values).
            moments_QMOM (array-like): Moments calculated using QMOM.
            moments_GQMOM (array-like): Moments calculated using GQMOM.
        """
        fig=plt.figure()
        ori_ax = fig.add_subplot(1,2,1)   
        rel_ax = fig.add_subplot(1,2,2)  
        # Calculate relative errors
        relative_error_QMOM = np.abs((moments_QMOM - moments) / moments)
        relative_error_GQMOM = np.abs((moments_GQMOM - moments) / moments)

        # Define the orders of moments
        orders = np.arange(len(moments))

        # Plot 1: Original values comparison
        ori_ax, fig = pt.plot_data(orders, moments, fig=fig, ax=ori_ax,
                                xlbl='Order of Moment',
                                ylbl='Moment Value',
                                lbl='Original Moments (True)',
                                clr='k',mrk='o')
        ori_ax, fig = pt.plot_data(orders, moments_QMOM, fig=fig, ax=ori_ax,
                                lbl='QMOM Moments',
                                clr='b',mrk='o')
        ori_ax, fig = pt.plot_data(orders, moments_GQMOM, fig=fig, ax=ori_ax,
                                lbl='GQMOM Moments',
                                clr='r',mrk='o')
        
        rel_ax, fig = pt.plot_data(orders, relative_error_QMOM, fig=fig, ax=rel_ax,
                                xlbl='Order of Moment',
                                ylbl='Relative Error',
                                lbl='Relative Error (QMOM)',
                                clr='b',mrk='o')
        rel_ax, fig = pt.plot_data(orders, relative_error_GQMOM, fig=fig, ax=rel_ax,
                                lbl='Relative Error (GQMOM)',
                                clr='r',mrk='o')
        ori_ax.grid('minor')
        ori_ax.set_yscale('log')
        rel_ax.grid('minor')
        plt.title('Comparison of Moments')
        plt.tight_layout()
        plt.legend()
        plt.show()

    def plot_NDF_comparison(self, x, NDF, NDF_QMOM, NDF_GQMOM):
        """
        Plot a visual comparison of QMOM and GQMOM NDF against original NDF.

        Parameters:
            NDF (array-like): Original NDF (true values).
            NDF_QMOM (array-like): NDF calculated using QMOM.
            NDF_GQMOM (array-like): NDF calculated using GQMOM.
        """
        fig=plt.figure()

        # Plot 1: Original values comparison
        ax, fig = pt.plot_data(x, NDF, fig=fig, ax=None,
                                xlbl='x',
                                ylbl='NDF',
                                lbl='Original(True)',
                                clr='k',mrk='o')
        ax, fig = pt.plot_data(x, NDF_QMOM, fig=fig, ax=ax,
                                lbl='QMOM',
                                clr='b',mrk='o')
        ax, fig = pt.plot_data(x, NDF_GQMOM, fig=fig, ax=ax,
                                lbl='GQMOM',
                                clr='r',mrk='o')
        
        ax.grid('minor')
        plt.title('Comparison of NDF')
        plt.tight_layout()
        plt.legend()
        plt.show()

    def plot_nodes_weights_comparision(self, x, NDF, nodes, weights, nodes_G, weights_G):
        fig=plt.figure()
        # ax1 = fig.add_subplot(1,3,1)   
        # ax2 = fig.add_subplot(1,3,2)
        # ax3 = fig.add_subplot(1,3,3)
        # Plot 1: Original values comparison
        ax1, fig = pt.plot_data(x, NDF, fig=fig, ax=None,
                                xlbl='x',
                                ylbl='NDF',
                                lbl='Original(True)',
                                clr='k',mrk='o')
        
        ax2 = ax1.twinx()
        ax2, fig = pt.plot_data(nodes, weights, fig=fig, ax=ax2,
                                lbl='QMOM',
                                clr='b',mrk='o')
        ax2, fig = pt.plot_data(nodes_G, weights_G, fig=fig, ax=ax2,
                                lbl='GQMOM',
                                clr='r',mrk='o')
        
        ax1.grid('minor')
        ax2.grid('minor')
        # ax3.grid('minor')
        plt.title('Comparison of nodes and weights')
        plt.tight_layout()
        plt.legend()
        plt.show()