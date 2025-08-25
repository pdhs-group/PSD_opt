# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 14:12:34 2021

@author: xy0264
"""

import general_scripts.global_constants as gc
import general_scripts.global_variables as gv
import numpy as np
# from memory_profiler import profile

# @profile
def check_del_t_2d():
    t=0
    red=0.5 # Reduction factor
    
    # Calculate concentration change matrix for FIRST timestep
    DN=get_deln_2d_geo(t,gv.N,gv.V,gv.V1,gv.V3,gv.F_M)
    
    # Due to numerical issues it is necessary to define a threshold for DN
    DN[np.abs(DN)<gc.THR_DN]=0
    
    # If timestep is too large, negative concentrations can occur! --> Adjust
    tmp_N=gv.N[:,:,t]+DN*gc.DEL_T
    DEL_T_0=gc.DEL_T
        
    while any(tmp_N[tmp_N<0]):
        print(f"""Current timestep {gc.DEL_T}s is too large. Adjusting..""")
        
        # Adjusting timestep
        gc.DEL_T=gc.DEL_T*red
        
        # Calculating new tmp_N for while criterion
        tmp_N=gv.N[:,:,t]+DN*gc.DEL_T
        
        # Break statement
        if DEL_T_0/gc.DEL_T>=red**-3:
            break
    
    # if gc.DEL_T!=DEL_T_0:
    #     # Adjusting timestep another time (?)
    #     gc.DEL_T=gc.DEL_T*0.5
            
    print(f"Final timestep is {gc.DEL_T}.")    
    # N matrix and NUM_T need adjustment (length of third axis)
    gc.NUM_T=round(gc.NUM_T*DEL_T_0/gc.DEL_T)
    
    # Initialize concentration matrix N
    N2=np.zeros((gc.NS+3,gc.NS+3,gc.NUM_T+1))
    N2[:,:,0]=gv.N[:,:,0]
    gv.N=N2
    
    del N2, DN, DEL_T_0, tmp_N, red, t
    
def check_del_t_3d(): 
    t=0
    red=0.5 # Reduction factor
    
    # Calculate concentration change matrix for FIRST timestep
    DN=get_deln_3d_geo(t,gv.N,gv.V,gv.V1,gv.V2,gv.V3,gv.F_M)
    
    # Due to numerical issues it is necessary to define a threshold for DN
    DN[np.abs(DN)<gc.THR_DN]=0
    
    # If timestep is too large, negative concentrations can occur! --> Adjust
    tmp_N=gv.N[:,:,:,t]+DN*gc.DEL_T
    DEL_T_0=gc.DEL_T
        
    while any(tmp_N[tmp_N<0]):
        print(f"""Current timestep {gc.DEL_T}s is too large. Adjusting..""")
        
        # Adjusting timestep
        gc.DEL_T=gc.DEL_T*0.5
        
        # Calculating new tmp_N for while criterion
        tmp_N=gv.N[:,:,:,t]+DN*gc.DEL_T
        
        # Break statement
        if DEL_T_0/gc.DEL_T>=red**-3:
            break
    
    # if gc.DEL_T!=DEL_T_0:
    #     # Adjusting timestep
    #     gc.DEL_T=gc.DEL_T*0.5
            
    print(f"Final timestep is {gc.DEL_T}.")    
    # N matrix and NUM_T need adjustment (length of third axis)
    gc.NUM_T=round(gc.NUM_T*DEL_T_0/gc.DEL_T)
    
    # Initialize concentration matrix N
    N2=np.zeros((gc.NS+3,gc.NS+3,gc.NS+3,gc.NUM_T+1))
    N2[:,:,:,0]=gv.N[:,:,:,0]
    gv.N=N2        
      
    del N2, DN, DEL_T_0, tmp_N, red, t
    
if __name__ == "__main__":
    pass 
        
        
        
        