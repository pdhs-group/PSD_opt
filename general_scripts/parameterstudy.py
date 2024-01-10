# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 09:01:02 2021

@author: xy0264
"""
import numpy as np
import os, sys
import openpyxl
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),".."))
from population import population as pop
from general_scripts.generate_psd import generate_psd_normal, find_x_f, full_psd
import general_scripts.global_constants as gc
import general_scripts.global_variables as gv
import time
from datetime import datetime
#from neural_net import myMLP as MLP
from extract_xls import extract_xls
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MaxAbsScaler
import gc as garbo
import tracemalloc

def modelparameterstudy(filename_mod=None, filename_exp=None, init=False, max_param=0, max_modparam=0, start_modp=0, ANN_flnm=None):
    
    # If filename_mod is not given, use default
    if filename_mod == None:
        filename_mod=os.path.join(os.path.dirname( __file__ ),'..',"data\\mod_DOE_data\\mod_DOE_210429.xlsx")
    
    # If filename_exp is not given, use default
    if filename_exp == None:
        filename_exp=os.path.join(os.path.dirname( __file__ ),'..',"data\\exp_data\\student_data_colletion_210423.xlsx")
    
    # Extract Model parameter data and get number of keys and datapoints
    mod_data = extract_xls(filename_mod)
    num_k = len(mod_data)
    num_modp = [len(mod_data[k]) for k in mod_data.keys()][0]
    
    # Extract number of process parameter combinations
    tmp = extract_xls(filename_exp)
    num_dp = [len(tmp[k]) for k in tmp.keys()][0]
    
    # Set number of modelcombinations to maxparam. If maxparam == 0 use all
    if max_modparam != 0 and max_modparam < num_modp:
        num_modp=max_modparam
        
    # Set up calculation by initializing global constants and variable (values changed later)
    if init:
        gv.initialize()
        gc.initialize()
    
    # Re-Define some parameters
    gc.REPORT=False    
        
    RMSE_vec=np.ones(num_modp)*1000
    T_mod_NM1_vec=np.zeros([num_modp,num_dp])  
    T_mod_NM2_vec=np.zeros([num_modp,num_dp])  
    T_mod_M_vec=np.zeros([num_modp,num_dp])    
    
    tme_stmp=f"_{datetime.now().time().hour:02d}{datetime.now().time().minute:02d}{datetime.now().time().second:02d}"
    exppth=os.path.join(os.path.dirname( __file__ ),'..',"export\\"+gc.FOLDERNAME+"\\")+'modelparameterstudy'+tme_stmp+'.npy'
        
    for i in range(start_modp,num_modp):
        
        # Set parameters
        gc.G=mod_data['G'][i]
        gc.NS=mod_data['NS'][i]
        gc.DEL_T=mod_data['DEL_T'][i]
        gc.S=mod_data['S'][i]
        gc.X_SEL=mod_data['X_SEL'][i]
        gc.Y_SEL=mod_data['Y_SEL'][i]
        gc.SIZEEVAL=mod_data['SIZEEVAL'][i]
        gc.CORR_PSI=mod_data['CORR_PSI'][i]
        gc.CORR_PSI_SIO2=mod_data['CORR_PSI'][i]
        gc.CORR_PSI_ZNO=mod_data['CORR_PSI'][i]
        gc.CORR_PSI_MAG=mod_data['CORR_PSI'][i]    
        gc.CORR_A_SIO2=mod_data['CORR_A_SIO2'][i] 
        gc.CORR_A_ZNO=mod_data['CORR_A_ZNO'][i]
        gc.CORR_A_MAG=mod_data['CORR_A_MAG'][i]  

        print("-- MODEL PARAMETER COMBINATION NUMBER ",i+1,"/",num_modp," --")
        print("G=",gc.G," | NS=", gc.NS," | DEL_T=", gc.DEL_T," | S=", gc.S," | X_SEL=", gc.X_SEL," | Y_SEL=", gc.Y_SEL," | SIZEEVAL=", gc.SIZEEVAL," | CORR_PSI_SIO2=", gc.CORR_PSI_SIO2," | CORR_PSI_ZNO=", gc.CORR_PSI_ZNO," | CORR_PSI_MAG=", gc.CORR_PSI_MAG," | CORR_A=", gc.CORR_A)
        
        # Calculate full parameterstudy
        #data, T_mod_NM1, T_mod_NM2, T_mod_M, _, RMSE_vec[i] = parameterstudy()
        data, T_mod_NM1, T_mod_NM2, T_mod_M, _, alphas, RMSE_vec[i] = parameterstudy(flnm_exp,max_param=max_param,ANN_flnm=ANN_flnm)
        
        # Save data in corresponding matrices (not directly done above, if dimensions missmatch)
        T_mod_NM1_vec[i,0:len(T_mod_NM1)]=T_mod_NM1
        T_mod_NM2_vec[i,0:len(T_mod_NM1)]=T_mod_NM2
        T_mod_M_vec[i,0:len(T_mod_NM1)]=T_mod_M
        
        # Extract experimental data 
        T_exp_NM1_vec=np.array(data['T_NM1 [-]'])*100
        T_exp_NM2_vec=np.array(data['T_NM2 [-]'])*100
        
        # Export results at every full calculation cycle
        results_dic={'mod_data':mod_data,'RMSE_vec':RMSE_vec,'T_mod_NM1_vec':T_mod_NM1_vec,\
                     'T_mod_NM2_vec':T_mod_NM2_vec,'T_mod_M_vec':T_mod_M_vec,\
                     'T_exp_NM1_vec':T_exp_NM1_vec,'T_exp_NM2_vec':T_exp_NM2_vec,\
                     'alphas':alphas,'exp_data':data}
        np.save(exppth,results_dic)    
     
    return results_dic

#@profile    
def parameterstudy(filename=None,init=False,max_param=0,ANN_flnm=None):
    
    # If filename is not given, use default
    if filename == None:
        filename=os.path.join(os.path.dirname( __file__ ),'..',"data\\exp_data\\student_data_colletion_210423.xlsx")
    
    # Extract data and get number of keys and datapoints
    data = extract_xls(filename)
    num_k = len(data)
    num_dp = [len(data[k]) for k in data.keys()][0]
    #print(num_dp)
    
    # Load ANN data if given
    if ANN_flnm!=None: ANN = np.load(ANN_flnm,allow_pickle=True).item()
            
    # Set up calculation by initializing global constants and variable (values changed later)
    if init:
        gv.initialize()
        gc.initialize()
        
    # Set number of datapoints to maxparam. If maxparam == 0 use all
    if max_param != 0 and max_param < num_dp:
        num_dp=max_param
    
    # Initialize result vectors and RMSE value
    T_mod_NM1=np.zeros(num_dp)
    T_mod_NM2=np.zeros(num_dp)
    T_mod_M=np.zeros(num_dp)
    SE_NM=np.zeros(num_dp)
    alphas=np.zeros((num_dp,4))
    
    # Save initial gc.DEL_T (may be adjusted for some and needs reset)
    DEL_T_0=gc.DEL_T
    
    # Go through all data points
    for i in range(num_dp):
        
        print("Calculating process-parameter combination no. ",i+1,"/",num_dp)
        
        # Reset default DEL_T
        gc.DEL_T=DEL_T_0
        
        # Pass the parameters of current datapoint to global variables
        gc.I=data['I [mol/L]'][i]*1e3                  # Ionic strength to mol/m³
        gc.NUM_T=round(data['t_A [min]'][i]*60/gc.DEL_T)  # Agglomeration time in timesteps
                
        # Get particle properties if entry of data['NM'] is not 0 
        # NM1
        if not data['NM1'][i] == 0:
            gc.PSI1, gc.V01, gc.DIST1, gc.R01, gc.A_NM1NM1 = \
                return_particle_properties(data['NM1'][i],data['pH [-]'][i],data['c_NM1 [g/L]'][i])
        # NM
        if not data['NM2'][i] == 0:
            gc.PSI2, gc.V02, gc.DIST2, gc.R02, gc.A_NM2NM2 = \
                return_particle_properties(data['NM2'][i],data['pH [-]'][i],data['c_NM2 [g/L]'][i])
        # M
        gc.PSI3, gc.V03, gc.DIST3, gc.R03, gc.A_MM = \
                return_particle_properties(data['M'][i],data['pH [-]'][i],data['c_M [g/L]'][i])
        
        # Use ANN to predict alpha prim if filename is given
        if ANN_flnm!=None:
            # Set EFFEVAL to 3 (=using reduced model, but not calculating alphas)
            gc.EFFEVAL=3
            gv.alpha_prim=np.zeros(4)
            
            # Set G to 1
            gc.G=1
            
            # Get Input vector (same as at training)
            X=np.zeros((1,3))
            if data['NM1'][i]=='SF800': X[0,0]=0
            if data['NM1'][i]=='ZNO': X[0,0]=1 
            if data['NM1'][i]=='SF300': X[0,0]=2
            X[0,1]=data['pH [-]'][i]    
            X[0,2]=np.log10(data['I [mol/L]'][i])
            
            # Transform inputs
            X=ANN['scaler'].transform(X)
            
            # Predict alphas
            pred=ANN['ANN'].predict(X.reshape(1,-1))
            
            # Re-transform if model was trained on log10 scale
            if ANN['transform_Y']: pred=10**pred
            gv.alpha_prim[0]=pred[0,0]
            gv.alpha_prim[1]=pred[0,1]
            gv.alpha_prim[2]=pred[0,1]
            gv.alpha_prim[3]=pred[0,2]            
            
            print(gv.alpha_prim)
            
        #print(gc.PSI1,gc.PSI3)
        # Calculate polulation balance. Depending on wheter data['NM2'] is 0 use 2D or 3D population balance
        if data['NM2'][i] == 0:
            T_mod_NM1[i],T_mod_M[i]=pop(2,"geo",False,True)
            # Calculate square error 
            SE_NM[i]=(T_mod_NM1[i]-data['T_NM1 [-]'][i]*100)**2
            alphas[i,:]=gv.alpha_prim
            print(f"Results: T1_mod={T_mod_NM1[i]:.1f} | T1_exp={data['T_NM1 [-]'][i]*100:.1f} | SE_NM={SE_NM[i]:.1f}")
            
        else: 
            T_mod_NM1[i],T_mod_NM2[i],T_mod_M[i],_,_=pop(3,"geo",False,True)
            # Calculate square error 
            SE_NM[i]=np.mean([(T_mod_NM1[i]-data['T_NM1 [-]'][i]*100)**2,(T_mod_NM2[i]-data['T_NM2 [-]'][i]*100)**2])
            print(f"Results: T1_mod={T_mod_NM1[i]:.1f} | T1_exp={data['T_NM1 [-]'][i]*100:.1f} | T2_mod={T_mod_NM2[i]:.1f} | T2_exp={data['T_NM2 [-]'][i]*100:.1f} | SE_NM={SE_NM[i]:.1f}")
        
    # Calculate Root Mean Square Error (RMSE)
    RMSE=np.sqrt(np.sum(SE_NM)/num_dp)       
    print("Root Mean Square Error (RMSE) = ",RMSE)
    
    del num_k, num_dp, DEL_T_0
    
    return data, T_mod_NM1, T_mod_NM2, T_mod_M, SE_NM, alphas, RMSE

def optimize_modparam(filename_exp=None, init=False, max_param=0):
        
    from scipy.optimize import shgo, basinhopping, minimize
    #import os
    
    # If filename_exp is not given, use default
    if filename_exp == None:
        filename_exp=os.path.join(os.path.dirname( __file__ ),'..',"data\\exp_data\\student_data_colletion_210423.xlsx")
             
    # Set up calculation by initializing global constants and variable (values changed later)
    if init:
        gv.initialize()
        gc.initialize()
    
    # Re-Define some parameters
    gc.REPORT=False 
    gc.NS=12
    gc.DEL_T=2
    gc.S=2.5
    gc.SIZEEVAL=2
            
    tme_stmp=f"_{datetime.now().time().hour:02d}{datetime.now().time().minute:02d}{datetime.now().time().second:02d}"
    exppth=os.path.join(os.path.dirname( __file__ ),'..',"export\\"+gc.FOLDERNAME+"\\")+'optimize_results_'+tme_stmp+'.npy'
        
    # Define INITIAL GUESS for optimization
    init_modparam=np.zeros(5)
    init_modparam[0]=0.493765     # G
    #init_modparam[1]=1     # X_SEL
    #init_modparam[2]=1.5   # Y_SEL
    init_modparam[1]=0.114    # CORR_PSI
    init_modparam[2]=0.00237    # CORR_A_SIO2
    init_modparam[3]=0.024762169    # CORR_A_ZNO
    init_modparam[4]=0.00526    # CORR_A_MAG
    
    # Define LOWER and UPPER BOUNDARIES for optimization
    bnds = ((0.1, 5),     # G
            #(0.1, 5),     # X_SEL             
            #(0.1, 5),     # Y_SEL 
            (0.01, 1.5),   # CORR_PSI 
            (0.001, 1),     # CORR_A_SIO2 
            (0.001, 1),     # CORR_A_ZNO 
            (0.001, 1))     # CORR_A_MAG 
    
    # Initialize temporary save dictionary and file
    res_optim_tmp={'modparam':np.zeros((1,len(init_modparam))),'RMSE':1000}
    np.save(gc.TMP_STR2,res_optim_tmp)
    
    # tracemalloc.start()
    # gc.TMP_NUM1=tracemalloc.take_snapshot()    
    
    # LOCAL optimization
    opt_mini={'disp':True}
    results=minimize(optim_fun,init_modparam,bounds=bnds,options=opt_mini)      
        
    # GLOBAL optimization
    opt_shgo={'maxtime':12*60*60} # 1h*60min/h*60s/min
    #results=shgo(optim_fun,bounds=bnds)#,options=opt_shgo) 
    #results=basinhopping(optim_fun,init_modparam) 
    
    np.save(exppth,results)
              
    return results

# This function takes in an arbitrary set of model parameters and return the corresponding RMSE value
#@profile 
def optim_fun(modparam):
    
    ## DEFINITION OF modparam:
    # modparam[0]: G
    # modparam[1]: CORR_PSI_SIO2 
    # modparam[2]: CORR_PSI_ZNO
    # modparam[3]: CORR_PSI_MAG
    # modparam[4]: CORR_A

    # Set parameters and reset gc.DEL_T
    gc.G=modparam[0]    
    gc.CORR_PSI=modparam[1]
    gc.CORR_A_SIO2=modparam[2]
    gc.CORR_A_ZNO=modparam[3]
    gc.CORR_A_MAG=modparam[4]
    gc.DEL_T=2
    
    print("G=",gc.G," | CORR_PSI=", gc.CORR_PSI," | CORR_A_SIO2=", gc.CORR_A_SIO2," | CORR_A_ZNO=", gc.CORR_A_ZNO," | CORR_A_MAG=", gc.CORR_A_MAG)
    
    # Only work if all parameters are > 0, otherwise set RMSE to 1000 and pass
    if any(t < 0 for t in modparam):
        RMSE=1000
    else:
        # Calculate full parameterstudy 
        _, _, _, _, _, _, RMSE = parameterstudy(gc.TMP_STR1,max_param=0)
        del _
    
    # Set RMSE to 1e3 if nan encountered (stability)
    if np.isnan(RMSE):
        RMSE=1000
    
    # Manual saving of current iteration data
    modsv=np.zeros((1,len(modparam)))
    modsv[-1,:]=modparam
    res_optim_tmp=np.load(gc.TMP_STR2,allow_pickle=True).item()
    res_optim_tmp['modparam']=np.append(res_optim_tmp['modparam'],modsv,axis=0) 
    res_optim_tmp['RMSE']=np.append(res_optim_tmp['RMSE'],RMSE)
    np.save(gc.TMP_STR2,res_optim_tmp)
    
    # Manual garbage collection (helps to free up memory?)
    garbo.collect()
    
    # Delete temporary variables (memory)
    del modsv, res_optim_tmp
    
    # gc.TMP_NUM2=tracemalloc.take_snapshot()  
    
    # top_stats = gc.TMP_NUM2.compare_to(gc.TMP_NUM1, 'lineno')

    # print("[ Top 3 differences ]")
    # for stat in top_stats[:3]:
    #     print(stat)
    
    # gc.TMP_NUM1=tracemalloc.take_snapshot() 
    
    return RMSE
    
def return_particle_properties(particle, pH, c_M):
    
    ## INPUTS
    # particle:     String containing particle name
    # pH:           pH Value in the experiment
    # c_M:          Mass concentration of particle in [g/L]
    
    ## OUTPUTS
    # psi:          Zeta potential of the particle in [V]
    # v:            Total volume concentration of the particle in [m³/m³]
    # dist:         Full filestring that can be used by initialize_psd()
    
    # Define density dictionary and calculate total volume concentration
    density={'SIO2MAG_MP_05':1800,'SF800':2650,'SF300':2650, 'ZNO': 5600,'PVC':1410} # in [g/L]
    v=c_M/density[particle]
    
    # Define Hamaker constant dictionary
    hamaker={'SIO2MAG_MP_05':4.6e-21,'SF800':1.02e-20,'SF300':1.02e-20, 'ZNO':1.89e-20,'PVC':7.8e-20} # in [J]
       
    # Define distribution filestring and retrieve R0
    dist=os.path.join(os.path.dirname( __file__ ),'..',"data\\PSD_data\\")+particle+'_pgv.npy'
    r0=np.load(dist,allow_pickle=True).item()['r0']
    
    # Read corresponding zeta dictionary and return values corresponding to pH from polyfit data
    # See ..data/zeta_data/extract_zeta_xls.py for more info
    psi_dic=np.load(os.path.join(os.path.dirname( __file__ ),'..',"data\\zeta_data\\")+particle+'_zeta.npy', allow_pickle=True).item()
    # If pH is outside of fitted range return message and set to closest in range point 
    # Reason: Fitting with polynomials is errous when extrapolating
    if pH<np.min(psi_dic['pH']):
        psi=np.polyval(psi_dic['pol'],np.min(psi_dic['pH']))*1e-3
        print("CARE: pH (=",pH,") is outside of measured Zeta - pH range for particle system", particle)
        print("Setting to closest in-range point ",np.min(psi_dic['pH']))
    elif pH>np.max(psi_dic['pH']):
        psi=np.polyval(psi_dic['pol'],np.max(psi_dic['pH']))*1e-3
        print("CARE: pH (=",pH,") is outside of measured Zeta - pH range for particle system", particle)
        print("Setting to closest in-range point ",np.max(psi_dic['pH']))
    else:
        psi=np.polyval(psi_dic['pol'],pH)*1e-3
    
    # Correct psi and A value with respective correction factor
    if particle == 'SF800' or particle == 'SF300':
        #psi=psi*gc.CORR_PSI_SIO2
        psi=psi*gc.CORR_PSI
        A=hamaker[particle]*gc.CORR_A_SIO2
    elif particle == 'ZNO':
        #psi=psi*gc.CORR_PSI_ZNO
        psi=psi*gc.CORR_PSI
        A=hamaker[particle]*gc.CORR_A_ZNO
    elif particle == 'SIO2MAG_MP_05':
        #psi=psi*gc.CORR_PSI_MAG
        psi=psi*gc.CORR_PSI
        A=hamaker[particle]*gc.CORR_A_MAG
    else:
        psi=psi*gc.CORR_PSI
        A=hamaker[particle]*gc.CORR_A
    
    return psi, v, dist, r0, A

def evaluate_modelparameterstudy(filename=None,results_dic=None):
    
    import matplotlib.pyplot as plt
    from general_scripts.KIT_cmap import c_KIT_green, c_KIT_red, c_KIT_blue, c_KIT_orange, c_KIT_purple
        
    # Define default filename if none is given
    if filename == None:
        filename=os.path.join(os.path.dirname( __file__ ),'..',"export\\"+gc.FOLDERNAME+"\\")+'modelparameterstudy.npy'
    
    if results_dic == None:
        # Read results file
        results_dic=np.load(filename,allow_pickle=True).item()
    
    # Find index with minimum RMSE value and thus the "best" combination
    i_min=np.argmin(results_dic['RMSE_vec'])
    print(f"The minimum RMSE index is {i_min} with a RMSE value of {results_dic['RMSE_vec'][i_min]:.3f}.")
    #print("The corresponding modelparameter combination is:")
    #print(f"G={results_dic['mod_data']['G'][i_min]} | NS={results_dic['mod_data']['NS'][i_min]} | DEL_T={results_dic['mod_data']['DEL_T'][i_min]} | S={results_dic['mod_data']['S'][i_min]} | X_SEL={results_dic['mod_data']['X_SEL'][i_min]} | Y_SEL={results_dic['mod_data']['Y_SEL'][i_min]} | SIZEEVAL={results_dic['mod_data']['SIZEEVAL'][i_min]} | CORR_PSI_SIO2={results_dic['mod_data']['CORR_PSI_SIO2'][i_min]} | CORR_PSI_ZNO={results_dic['mod_data']['CORR_PSI_ZNO'][i_min]} | CORR_A={results_dic['mod_data']['CORR_A'][i_min]}")
        
    # Create sub-array depending on NM1 and NM2
    T_NM1_SF800=np.array([])
    T_NM1_exp_SF800=np.array([])
    T_NM1_ZNO=np.array([])
    T_NM1_exp_ZNO=np.array([])
    T_NM1_PVC=np.array([])
    T_NM1_exp_PVC=np.array([])
    T_NM1_SF300=np.array([])
    T_NM1_exp_SF300=np.array([])
    T_NM2_ZNO=np.array([])
    T_NM2_exp_ZNO=np.array([])
    T_NM2_PVC=np.array([])
    T_NM2_exp_PVC=np.array([])
    for i in range(len(results_dic['exp_data']['NM1'])):
        if results_dic['exp_data']['NM1'][i]=='SF800':
            T_NM1_SF800=np.append(T_NM1_SF800,results_dic['T_mod_NM1_vec'][i_min,i])
            T_NM1_exp_SF800=np.append(T_NM1_exp_SF800,results_dic['T_exp_NM1_vec'][i])
        if results_dic['exp_data']['NM1'][i]=='ZNO':
            T_NM1_ZNO=np.append(T_NM1_ZNO,results_dic['T_mod_NM1_vec'][i_min,i])
            T_NM1_exp_ZNO=np.append(T_NM1_exp_ZNO,results_dic['T_exp_NM1_vec'][i])
        if results_dic['exp_data']['NM1'][i]=='PVC':
            T_NM1_PVC=np.append(T_NM1_PVC,results_dic['T_mod_NM1_vec'][i_min,i])
            T_NM1_exp_PVC=np.append(T_NM1_exp_PVC,results_dic['T_exp_NM1_vec'][i])
        if results_dic['exp_data']['NM1'][i]=='SF300':
            T_NM1_SF300=np.append(T_NM1_SF300,results_dic['T_mod_NM1_vec'][i_min,i])
            T_NM1_exp_SF300=np.append(T_NM1_exp_SF300,results_dic['T_exp_NM1_vec'][i])
        if results_dic['exp_data']['NM2'][i]=='ZNO':
            T_NM2_ZNO=np.append(T_NM2_ZNO,results_dic['T_mod_NM2_vec'][i_min,i])
            T_NM2_exp_ZNO=np.append(T_NM2_exp_ZNO,results_dic['T_exp_NM2_vec'][i])
        if results_dic['exp_data']['NM2'][i]=='PVC':
            T_NM2_PVC=np.append(T_NM2_PVC,results_dic['T_mod_NM2_vec'][i_min,i])
            T_NM2_exp_PVC=np.append(T_NM2_exp_PVC,results_dic['T_exp_NM2_vec'][i])
        
    # Scatter plot at minimum index
    # General plot setup
    scl=1.2
    plt.rc('mathtext', fontset='cm')
    plt.rc('font', family='Latin Modern Roman')
    plt.rc('xtick', labelsize=10*scl)
    plt.rc('ytick', labelsize=10*scl)
    plt.rc('axes', labelsize=12*scl, linewidth=0.5*scl)
    plt.rc('legend', fontsize=10*scl, handlelength=3*scl)
    
    # Close all and setup figure
    plt.close('all')
    fig, ax = plt.subplots(1,2,figsize=np.array([6.4, 3.2])*2) 
    
    # NM1
    #ax[0].scatter(results_dic['T_exp_NM1_vec'],results_dic['T_mod_NM1_vec'][i_min,:],label='NM1',color='k') 
    ax[0].scatter(T_NM1_exp_SF800,T_NM1_SF800,label='SF800',color=c_KIT_green,marker='o') 
    ax[0].scatter(T_NM1_exp_SF300,T_NM1_SF300,label='SF300',color=c_KIT_orange,marker='*') 
    ax[0].scatter(T_NM1_exp_ZNO,T_NM1_ZNO,label='ZNO',color=c_KIT_blue,marker='s') 
    ax[0].scatter(T_NM1_exp_PVC,T_NM1_PVC,label='PVC',color=c_KIT_red,marker='^') 
    ax[0].set_xlim(left=-5,right=105)
    ax[0].set_ylim(bottom=-5,top=105)
    ax[0].set_title('NM1')
    ax[0].plot([0, 1], [0, 1], transform=ax[0].transAxes, color='k', linestyle='-.', linewidth=0.75)
    ax[0].set_xlabel('Experimental values $T_{exp,NM1}$ / $\%$')
    ax[0].set_ylabel('Calculated values $T_{mod,NM1}$ / $\%$')
    ax[0].legend(loc='upper left')
    ax[0].grid(True)
    
    # NM2
    #ax[1].scatter(results_dic['T_exp_NM2_vec'],results_dic['T_mod_NM2_vec'][i_min,:],label='NM2',color='k') 
    ax[1].scatter(T_NM2_exp_ZNO,T_NM2_ZNO,label='ZNO',color=c_KIT_blue,marker='s') 
    ax[1].scatter(T_NM2_exp_PVC,T_NM2_PVC,label='PVC',color=c_KIT_red,marker='^') 
    ax[1].set_xlim(left=-5,right=105)
    ax[1].set_ylim(bottom=-5,top=105)
    ax[1].set_title('NM2')
    ax[1].plot([0, 1], [0, 1], transform=ax[1].transAxes, color='k', linestyle='-.', linewidth=0.75)
    ax[1].set_xlabel('Experimental values $T_{exp,NM2}$ / $\%$')
    ax[1].set_ylabel('Calculated values $T_{mod,NM2}$ / $\%$')
    ax[1].legend(loc='upper left')
    ax[1].grid(True)
    
    return results_dic

# %% MAIN    
if __name__ == '__main__':
    
    gv.initialize()
    gc.initialize()
    
    #psi, v, dist, r0, A = return_particle_properties('SIO2MAG_MP_05',12,1)  
    
    # Calculate modelparameterstudy
    flnm_mod=os.path.join(os.path.dirname( __file__ ),'..',"data\\mod_DOE_data\\mod_DOE_single_comb.xlsx")
    #flnm_mod=os.path.join(os.path.dirname( __file__ ),'..',"data\\mod_DOE_data\\mod_DOE_210908.xlsx")
    
    #flnm_ANN=os.path.join(os.path.dirname( __file__ ),'..',"data\\neural_net_data\\ANN_bayesian_210923.npy")
    flnm_ANN=os.path.join(os.path.dirname( __file__ ),'..',"data\\neural_net_data\\ANN_OZ_211103.npy")
    
    #flnm_exp=os.path.join(os.path.dirname( __file__ ),'..',"data\\exp_data\\data_set_OZ_modeltest_211029.xlsx")
    #flnm_exp=os.path.join(os.path.dirname( __file__ ),'..',"data\\exp_data\\student_data_collection_ANN_210913.xlsx")
    flnm_exp=os.path.join(os.path.dirname( __file__ ),'..',"data\\exp_data\\data_set_optimize_all_211124.xlsx")
    #flnm_exp=os.path.join(os.path.dirname( __file__ ),'..',"data\\exp_data\\data_set_ESM_2_3_Stoff_210927.xlsx")
    #flnm_exp=os.path.join(os.path.dirname( __file__ ),'..',"data\\exp_data\\data_single_test_210903.xlsx")
    
    #results_dic=modelparameterstudy(flnm_mod,flnm_exp,init=True,max_param=0,max_modparam=256,start_modp=0,ANN_flnm=flnm_ANN)
    #results_dic=modelparameterstudy(flnm_mod,flnm_exp,init=True,max_param=0,max_modparam=256,start_modp=0)
    
    # Run Optimizer, set flnm_exp to TMP_STR1 and define output path in TMP_STR2
    gc.TMP_STR1=flnm_exp
    name_marker='501'
    gc.TMP_STR2=os.path.join(os.path.dirname( __file__ ),'..',f"export\\210414_default\\tmp_optimizer_{name_marker}.npy")
    
    #res=optimize_modparam(flnm_exp, init=False, max_param=0)
    
    # Evaluate modelparameterstudy
    #exppth=os.path.join(os.path.dirname( __file__ ),'..',"export\\"+gc.FOLDERNAME+"\\")+'modelparameterstudy_evtest.npy'
    #exppth=os.path.join(os.path.dirname( __file__ ),'..',"export\\"+gc.FOLDERNAME+"\\")+'modelparameterstudy_104904.npy'
    #exppth=os.path.join(os.path.dirname( __file__ ),'..',"export\\"+gc.FOLDERNAME+"\\")+'modelparameterstudy_123231.npy'
    #exppth=os.path.join(os.path.dirname( __file__ ),'..',"export\\"+gc.FOLDERNAME+"\\")+'modelparameterstudy_132438.npy'
    exppth=os.path.join(os.path.dirname( __file__ ),'..',"results\\211123_Optimierung\\")+'modelparameterstudy_085702.npy'
    #exppth=os.path.join(os.path.dirname( __file__ ),'..',"results\\210906_optimizer_brute_Ns1\\")+'modelparameterstudy_093851.npy'
    
    results_dic = evaluate_modelparameterstudy(exppth)
    #results_dic = evaluate_modelparameterstudy(exppth,results_dic)