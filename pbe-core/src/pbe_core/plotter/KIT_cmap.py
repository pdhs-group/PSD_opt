"""
Karlsruhe Institute of Technology (KIT) colormap for consistent plots

@author: Frank Rhein
"""
import numpy as np
from matplotlib.colors import ListedColormap

c_KIT_green=np.array([0,150,130,255])/255  # Color 1 = KIT Green
c2=np.array([0,0,0,1])              # Color 2 = black 
c_KIT_red=np.array([162,34,35,255])/255  # Color 3 = KIT Red  
c4=np.array([1,1,1,1])              # Color 4 = white
c_KIT_blue=np.array([70,100,170,255])/255  # Color 3 = KIT Blue
c_KIT_orange=np.array([223,155,27,255])/255  # Color 3 = KIT orange
c_KIT_purple=np.array([163,16,124,255])/255  # Color 3 = KIT purple

fac=np.linspace(0, 1, 256)
fac3_1=np.concatenate((np.linspace(1, 0, 128),np.zeros(128)))
fac3_2=np.concatenate((np.linspace(0, 1, 128),np.linspace(1, 0, 128)))
fac3_3=np.concatenate((np.zeros(128),np.linspace(0, 1, 128)))
cmap_gb=np.zeros((256,4))
cmap_gw=np.zeros((256,4))
cmap_gr=np.zeros((256,4))
cmap_rb=np.zeros((256,4))
cmap_rw=np.zeros((256,4))
cmap_bgw=np.zeros((256,4))
cmap_bgr=np.zeros((256,4))
cmap_brw=np.zeros((256,4))
cmap_bbw=np.zeros((256,4))

for i in range(len(fac)):
    cmap_gb[i]=fac[i]*c_KIT_green+(1-fac[i])*c2
    cmap_gw[i]=fac[i]*c_KIT_green+(1-fac[i])*c4
    cmap_gr[i]=fac[i]*c_KIT_green+(1-fac[i])*c_KIT_red
    cmap_rb[i]=fac[i]*c_KIT_red+(1-fac[i])*c2
    cmap_rw[i]=fac[i]*c_KIT_red+(1-fac[i])*c4
    cmap_bgw[i]=fac3_1[i]*c2+fac3_2[i]*c_KIT_green+fac3_3[i]*c4    
    cmap_bgr[i]=fac3_1[i]*c_KIT_blue+fac3_2[i]*c_KIT_green+fac3_3[i]*c_KIT_red
    cmap_brw[i]=fac3_1[i]*c2+fac3_2[i]*c_KIT_red+fac3_3[i]*c4
    cmap_bbw[i]=fac3_1[i]*c2+fac3_2[i]*c_KIT_blue+fac3_3[i]*c4
    

# cmap_gw2=np.flip(cmap_gw,axis=0)
# slc=slice(0,-1,2)
# cmap_bgw[0:128]=cmap_gb[slc] 
# cmap_bgw[128::]=cmap_gw2[slc]   
KIT_green_black = ListedColormap(cmap_gb)
KIT_green_white = ListedColormap(cmap_gw)
KIT_green_red = ListedColormap(cmap_gr)
KIT_red_black = ListedColormap(cmap_rb)
KIT_red_white = ListedColormap(cmap_rw)
KIT_black_green_white = ListedColormap(cmap_bgw)
KIT_blue_green_red = ListedColormap(cmap_bgr)
KIT_black_red_white = ListedColormap(cmap_brw)
KIT_black_blue_white = ListedColormap(cmap_bbw)