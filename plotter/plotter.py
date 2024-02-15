# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 07:05:56 2021

@author: Frank Rhein
"""
import matplotlib.pyplot as plt

print('Imported correct plotter module (08.01.2024)') 
   
# ----------------------------------
# Define Plot defaults
# ----------------------------------
# scl_a4 define scaling of given figuresize 
    # 2: half page figure. Generate additional margin for legend.
    # 1: full page figure. Generate additional margin for legend.
# page_lnewdth_cm: Linewidth of document in cm
# scl: Additional font scaling
# fnt: Font type used in plot
# figsze: Figure size in inches
# frac_lnewdth: Additional scaling option to width = frac_lnewith*document_linewidth
# mrksze: Markersize
# lnewdth: Linewidth (of lines used in plot)
# use_locale: If True use local number format
def plot_init(scl_a4=1,page_lnewdth_cm=16.5,scl=1,fnt='Arial',figsze=[6.4,4.8],
              frac_lnewdth=0.6,mrksze=6,lnewdth=1.5,use_locale=False, fontsize = 10,
              labelfontsize=9, tickfontsize=8):
    
    # --- Initialize defaults ---
    plt.rcdefaults()
    
    # --- Scale figure ---
    # 2: Half page figure
    if scl_a4==2:     
        fac=scl*page_lnewdth_cm/(2.54*figsze[0]*2) #2.54: cm --> inch
        figsze=[figsze[0]*fac,figsze[1]*fac]
    
    # 2: Full page figure
    elif scl_a4==1:
        fac=scl*page_lnewdth_cm/(2.54*figsze[0]) #2.54: cm --> inch
        figsze=[figsze[0]*fac,figsze[1]*fac]
    
    # 3: Scaling for presentations (OLD)
    elif scl_a4==3:
        fac=scl*page_lnewdth_cm/(2.54*figsze[0]) #2.54: cm --> inch
        figsze=[figsze[0]*fac,figsze[1]*fac]
        scl=1.6
        
    # 4: Scaling for variable fraction of linewidth frac_lnewdth
    elif scl_a4==4:
        fac=scl*frac_lnewdth*page_lnewdth_cm/(2.54*figsze[0])
        figsze=[figsze[0]*fac,figsze[1]*fac]
    
    # --- Adjust legend ---
    plt.rc('legend',fontsize=fontsize*scl,fancybox=True, shadow=False,edgecolor='k',
           handletextpad=0.2,handlelength=1,borderpad=0.2,
           labelspacing=0.2,columnspacing=0.2)
    
    # --- General plot setup ---
    plt.rc('mathtext', fontset='cm')
    plt.rc('font', family=fnt)
    plt.rc('xtick', labelsize=tickfontsize*scl)
    plt.rc('ytick', labelsize=tickfontsize*scl)
    plt.rc('axes', labelsize=labelfontsize*scl, linewidth=0.5*scl, titlesize=labelfontsize*scl)
    plt.rc('legend', fontsize=fontsize*scl)
    plt.rc('axes', axisbelow=True) # Grid lines in Background
    plt.rcParams['lines.markersize']=mrksze
    plt.rcParams['hatch.linewidth']=lnewdth/2
    plt.rcParams['lines.linewidth']=lnewdth     
    plt.rcParams['figure.figsize']=figsze
    
    if use_locale: plt.rc('axes.formatter',use_locale=True)

# ----------------------------------
# Standard plot function
# ----------------------------------    
# x: x-Data
# y: y-Data
# err: (optional) Error data
# fig: (optional) Plot in given fig. Create new if None
# ax: (optional) Plot in given ax. Create new in None
# plt_type: Type of Plot
    # Default: Points with lines
    # 'scatter': Scatter without lines
    # 'bar': Bar plot
    # 'line': Line plot   
# lbl: (optional) Label of given Dataset (legend entry)
# xlbl: (optional) Label of x-axis
# ylbl: (optional) Label of y-axis
# mrk: (optional) Marker type, defaul 'o'
# lnstyle: (optional) Linestyle, default '-'
# clr: (optional) Color of plot, default 'k'
# tit: (optional) Title of plot
# grd: (optional) Set grid, default 'major. For no grid use None
# grd_ax: (optional) Define axis for grid, defaul 'both'
# leg: (optional) Bool. Define if legend is plotted, default True
# leg_points_only: (optional) Bool. If True only marker are plottet in legend 
# barwidth: (optional) Width of bar plot
# hatch: (optional) Hatch of bar plot
# alpha: (optional) Alpha of plot
# err_clr: (optional) Color of error bars, If None (default) use plot color
# zorder: (optional) Z-Order of plot. Higher values plotted above lower values
# mrkedgecolor: (optional) Edgecolor of marker
# mrkedgewidth: (optional) Width of marker edge
def plot_data(x,y,err=None,fig=None,ax=None,plt_type=None,lbl=None,xlbl=None,ylbl=None,
              mrk='o',lnstyle='-',clr='k',tit=None,grd='major',grd_ax='both',leg=True,
              leg_points_only=False,barwidth=0.5,hatch=None,alpha=1,err_clr=None,zorder=None,
              mrkedgecolor=None,mrkedgewidth=0.5, err_ax='y'):

    # --- If fig is not given by user: create new figure --- 
    if fig == None:
        fig=plt.figure()
    
    # --- If ax is not given create new axis on figure (only reasonable if fig==None also) ---
    if fig == None or ax == None:    
        ax=fig.add_subplot(1,1,1)
    
    # --- Plot data according to plt_type ---
    # --- Scatter ---
    if plt_type == 'scatter':  
        if mrkedgecolor == None:
            ax.scatter(x,y,label=lbl,marker=mrk,color=clr,zorder=zorder,alpha=alpha)
        else:
            ax.scatter(x,y,label=lbl,marker=mrk,color=clr,zorder=zorder,edgecolor=mrkedgecolor,linewidths=mrkedgewidth,alpha=alpha)
    
    # --- Bar ---
    elif plt_type == 'bar':    
        ax.bar(x,y,width=barwidth,label=lbl,color=clr,edgecolor='k',alpha=alpha,hatch=hatch,zorder=zorder,linewidth=plt.rcParams['hatch.linewidth'])
    
    # --- Line ---
    elif plt_type == 'line':
        # NOTE: If only line is plotted increase default linewidth by 50%
        ax.plot(x,y,label=lbl,linestyle=lnstyle,color=clr,linewidth=1.5*plt.rcParams['lines.linewidth'],alpha=alpha,zorder=zorder)
    
    # --- Default: Marker and Lines ---
    else:
        # --- Plot scatter first to show up in legend ---
        if leg_points_only: 
            if mrkedgecolor == None:
                ax.scatter(x,y,label=lbl,marker=mrk,color=clr,zorder=zorder,alpha=alpha)
                ax.plot(x,y,marker=mrk,linestyle=lnstyle,color=clr,zorder=zorder,alpha=alpha)
            else:
                ax.scatter(x,y,label=lbl,marker=mrk,color=clr,zorder=zorder,edgecolor=mrkedgecolor,linewidths=mrkedgewidth,alpha=alpha)
                ax.plot(x,y,marker=mrk,linestyle=lnstyle,color=clr,zorder=zorder,mec=mrkedgecolor,mew=mrkedgewidth,alpha=alpha)
        else:
            if mrkedgecolor == None:
                ax.plot(x,y,label=lbl,marker=mrk,linestyle=lnstyle,color=clr,zorder=zorder,alpha=alpha)
            else:
                ax.plot(x,y,label=lbl,marker=mrk,linestyle=lnstyle,color=clr,zorder=zorder,mec=mrkedgecolor,mew=mrkedgewidth,alpha=alpha)

    # --- Plot errorbars if error is given, if no err_color is given use plot color ---
    if err_clr == None: err_clr=clr
    if err is not None:
        if err_ax =='y':
            if plt_type == 'bar':  
                ax.errorbar(x,y,yerr=err,fmt='none',color=err_clr,capsize=plt.rcParams['lines.markersize']-2,alpha=0.5,zorder=99)
            else:
                ax.errorbar(x,y,yerr=err,fmt='none',color=err_clr,capsize=plt.rcParams['lines.markersize']-2,alpha=0.5,zorder=0)
        else:
            if plt_type == 'bar':  
                ax.errorbar(x,y,xerr=err,fmt='none',color=err_clr,capsize=plt.rcParams['lines.markersize']-2,alpha=0.5,zorder=99)
            else:
                ax.errorbar(x,y,xerr=err,fmt='none',color=err_clr,capsize=plt.rcParams['lines.markersize']-2,alpha=0.5,zorder=0)
        
    # --- Set labels, title and grid if given ---
    if xlbl != None: ax.set_xlabel(xlbl)
    if ylbl != None: ax.set_ylabel(ylbl)
    if tit != None: ax.set_title(tit)
    if grd!=None: ax.grid(True,which=grd,axis=grd_ax,alpha=0.5)
    if leg: ax.legend()
    
    # --- return ax and fig ---
    return ax, fig

# ----------------------------------
# Calculate and plot 2D histogram
# ----------------------------------    
# x: x-Data
# y: y-Data
# w: (optional) weights
# bins: (optional) Tuple with bins in (x,y) direction
# scale: (optional) Tubple with scale of (x,y) data ['lin'/'log']
# fig: (optional) Plot in given fig. Create new if None
# ax: (optional) Plot in given ax. Create new in None
# xlbl: (optional) Label of x-axis
# ylbl: (optional) Label of y-axis
# clr: (optional) Colormap of plot
# tit: (optional) Title of plot
# grd: (optional) Set grid
# norm: (optional) If True normalize histogram (on np.sum(H))
# colorbar: (optional) If True plot colorbar
# cblbl: (optional) Label for Colorbar
# only_calc: (optional) Only calculate bins and data without plotting
def plot_2d_hist(x,y,w=None,bins=(10,10),scale=('lin','lin'),fig=None,ax=None,xlbl=None,
                 ylbl=None, clr='viridis',tit=None,grd=True, norm=True, colorbar=True, cblbl='',
                 only_calc=False, scale_hist='lin', hist_thr=1e-6):
    
    import numpy as np
    from mpl_toolkits.axes_grid1 import make_axes_locatable    
    import matplotlib.colors as colors

    # --- Calculate histogram data
    H, xe, ye = np.histogram2d(x, y, bins=bins, weights=w)
    
    if scale[0] == 'log':
        logbins_x = np.logspace(np.log10(xe[0]),np.log10(xe[-1]),len(xe))
        
        if scale[1] == 'log': 
            logbins_y = np.logspace(np.log10(ye[0]),np.log10(ye[-1]),len(ye))
            H, xe, ye = np.histogram2d(x, y, bins=(logbins_x, logbins_y), weights=w)
            
        else:
            H, xe, ye = np.histogram2d(x, y, bins=(logbins_x, ye), weights=w)
    
    # np.histogram requires transposition (see docs)
    H = H.T
    
    if norm: H = H/np.sum(H)
    
    # --- Plot Histogram ---    
    if only_calc:
        fig, ax, cb = None, None, None
        
    else:
        X, Y = np.meshgrid(xe, ye)
        
        if grd: 
            ecl = 'k'
        else:
            ecl = None
            
        # --- If fig is not given by user: create new figure --- 
        if fig == None:
            fig=plt.figure()
        
        # --- If ax is not given create new axis on figure (only reasonable if fig==None also) ---
        if fig == None or ax == None:    
            ax=fig.add_subplot(1,1,1)
        
        if scale_hist == 'log':
            # H == 0 is not allowed. set to hist_thr*H.max() 
            H[H==0] = hist_thr*H.max()
            
            cp = ax.pcolormesh(X, Y, H, cmap=clr, edgecolor=ecl, antialiased=True, linewidth=0.1,
                               norm=colors.LogNorm(vmin=H.min(), vmax=H.max()))
        else:
            cp = ax.pcolormesh(X, Y, H, cmap=clr, edgecolor=ecl, antialiased=True, linewidth=0.1)
        
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cb = plt.colorbar(cp,cax)
            cb.set_label(cblbl)
        else:
            cb = None
            
        # --- Set labels, title and grid if given ---
        if scale[0] == 'log': ax.set_xscale('log')
        if scale[1] == 'log': ax.set_yscale('log')
        if xlbl != None: ax.set_xlabel(xlbl)
        if ylbl != None: ax.set_ylabel(ylbl)
        if tit != None: ax.set_title(tit)
        
    # --- return ax and fig ---
    return ax, fig, cb, H, xe, ye

# ----------------------------------
# Export current plot
# ----------------------------------    
# filename: Path for export. File extension determines format! (.pdf / .png)
# squeeze: Bool. Define if tight layout is used or not, default True
# dpi: DPI value for export. Only relevant for picture formats like .png / .jpg        
def plot_export(filename,squeeze=True,dpi=1000,pad_inch=False):
    
    if squeeze: 
        bb='tight' 
    else: 
        bb=None
    
    if pad_inch: 
        plt.savefig(filename,dpi=dpi,bbox_inches=bb,pad_inches = 0)
    else:
        plt.savefig(filename,dpi=dpi,bbox_inches=bb)
        
# ----------------------------------
# Close all plots (no import of matplotlib required)
# ----------------------------------  
def close():
    plt.close('all')
    

          
        
        
        
        
        
        