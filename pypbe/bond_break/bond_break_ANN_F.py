# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:34:57 2024

@author: px2030
"""
import time
import numpy as np
import os,sys
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),"../.."))
from pypbe.bond_break.bond_break_post import calc_int_BF
from pypbe.utils.func.jit_pop import lam, lam_2d, heaviside
import pypbe.bond_break.ANN_model_func as model_func
## external package
import math
import random
import pickle
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
## for plotter
import matplotlib.pyplot as plt
import pypbe.utils.plotter.plotter as pt
from pypbe.utils.plotter.KIT_cmap import c_KIT_green, c_KIT_red, c_KIT_blue, KIT_black_green_white

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

class ANN_bond_break():
    def __init__(self,dim,NS,S,global_seed):
        self.pth = os.path.dirname( __file__ )
        self.directory = os.path.join(self.pth,'../../tests/simulation_data')
        self.test_directory = os.path.join(self.pth,'../../tests/test_data')
        self.path_scaler = os.path.join(self.pth,'Inputs_scaler.pkl')
        self.path_all_data = os.path.join(self.pth,'output_data_volX1.pkl')
        
        self.global_seed = global_seed
        self.dim = dim
        self.NS = NS
        self.S = S
        self.N_GRIDS = 200
        self.N_FRACS = 100
        self.use_custom_loss = True
        self.save_model = True
        self.save_validate_results = True
        self.print_status = True
        
        ## Super parameter of model and training
        if dim == 1:
            self.init_neurons = 128
            self.num_layers = 2
        else:
            self.init_neurons = 256
            self.num_layers = 3
        self.dropout_rate = 0.5
        self.l1_factor = 0.001
        self.weight_loss_FRAG_NUM = 0.001
        self.batch_size = 32
        ## optimizer_type = 'adam' / 'sgd' / 'rmsprop'
        self.optimizer_type = 'adam'
        self.learning_rate = 0.0001
        
        self.generate_grid()
        
        ## Setting the global random seed
        if global_seed != 0:
            np.random.seed(global_seed)
            tf.random.set_seed(global_seed)
            random.seed(global_seed)
        
    def load_all_data(self, directory):
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]
        data_num = len(files)
        all_Inputs = np.zeros((data_num, 7))
        F_norm = []
        num = 0
        for i, file in enumerate(files):
            filename = os.path.basename(file)
            parsed_variables = np.array(filename[:-4].split('_'),dtype=float)
            F_tem = np.load(file,allow_pickle=True)
            shape = F_tem.shape[0]
            F_norm_tem = np.zeros((shape,3))
            A = parsed_variables[0]
            # if NO_FRAG_tem == NO_FRAG:
            #     continue
            # NO_FRAG_tem = NO_FRAG
            ## Normalize fragments volume
            F_norm_tem[:,0] = F_tem[:,0] * F_tem[:,1] / A
            F_norm_tem[:,1] = F_tem[:,0] * F_tem[:,2] / A
            F_norm_tem[:,2] = F_tem[:,3]
            all_Inputs[i,:] = parsed_variables
            F_norm.append(F_norm_tem)
            num += 1
            # if num > 20:
            #     break
            
        return all_Inputs, F_norm, data_num
    
    # %% DATA PROCESSING
    def generate_grid(self):
        e1 = np.zeros(self.NS+1)
        e2 = np.zeros(self.NS+1)
        ## Because of the subsequent CAT processing, x and y need to be expanded 
        ## outward by one unit with value -1
        self.x = np.zeros(self.NS+1)-1
        self.y = np.zeros(self.NS+1)-1
        e1[0] = -1
        e2[0] = -1
        for i in range(self.NS):
            e1[i+1] = self.S**(i)
            e2[i+1] = self.S**(i)
            self.x[i] = (e1[i] + e1[i+1]) / 2
            self.y[i] = (e2[i] + e2[i+1]) / 2
        e1[:] /= self.x[-2] 
        e2[:] /= self.y[-2] 
        self.x[:] /= self.x[-2] 
        self.y[:] /= self.y[-2] 
        B_c = np.zeros((self.NS+1,self.NS+1)) 
        v1 = np.zeros((self.NS+1,self.NS+1))
        v2 = np.zeros((self.NS+1,self.NS+1))
        M1_c = np.zeros((self.NS+1,self.NS+1))
        M2_c = np.zeros((self.NS+1,self.NS+1))
        return B_c,v1,v2,M1_c,M2_c,e1,e2
    
    def Outputs_on_grid(self,all_Inputs, F_norm, data_num,
                        B_c,v1,v2,M1_c,M2_c,e1,e2):
        Inputs_1d = []
        Inputs_2d = []
        Outputs_1d = []
        Outputs_1d_E = []
        Outputs_2dX1 = []
        Outputs_2dX2 = []
        Outputs_2d_E = []
        all_prob = np.zeros(data_num)
        all_x_mass = np.zeros(data_num)
        all_y_mass = np.zeros(data_num)
        for i in range(data_num):
            X1 = all_Inputs[i,1]
            NO_FRAG = all_Inputs[i,2]
            F_norm_tem = F_norm[i]
            if X1 == 1:
                tem_Outputs, all_prob[i], all_x_mass[i] = self.direct_psd_1d(
                    F_norm_tem[:,0],NO_FRAG,B_c,v1,M1_c,e1,self.x)
                Outputs_1d.append(tem_Outputs)
                tem_Inputs = [all_Inputs[i,j] for j in range(7) if j == 0 or j == 2 or j == 6]
                Inputs_1d.append(tem_Inputs)
                mean_energy = F_norm_tem[:,2].mean()
                Outputs_1d_E.append(mean_energy)
            else:
                tem_OutputsX1,tem_OutputsX2, all_prob[i], all_x_mass[i], all_y_mass[i] = self.direct_psd_2d(
                    F_norm_tem,NO_FRAG,X1,B_c,v1,v2,M1_c,M2_c,e1,e2,self.x,self.y)   
                Outputs_2dX1.append(tem_OutputsX1)
                Outputs_2dX2.append(tem_OutputsX2)
                Inputs_2d.append(all_Inputs[i,:])
                mean_energy = F_norm_tem[:,2].mean()
                Outputs_2d_E.append(mean_energy)

        scaler_1d = StandardScaler()
        scaler_2d = StandardScaler()
        with open(self.path_scaler,'wb') as file:
            pickle.dump((scaler_1d,scaler_2d), file)

        Inputs_1d_scaled = scaler_1d.fit_transform(Inputs_1d)
        Inputs_2d_scaled = scaler_2d.fit_transform(Inputs_2d)
        all_data = [[np.array(Inputs_1d),np.array(Outputs_1d),np.array(Inputs_1d_scaled),np.array(Outputs_1d_E)],
                    [np.array(Inputs_2d),np.array(Outputs_2dX1),np.array(Outputs_2dX2),np.array(Inputs_2d_scaled),np.array(Outputs_2d_E)]]
    
        with open(self.path_all_data, 'wb') as file:
            pickle.dump((all_data, all_prob, all_x_mass, all_y_mass), file)      
        return all_data, all_prob, all_x_mass, all_y_mass
    
    def direct_psd_1d(self,F_norm,NO_FRAG,B_c_tem,v1_tem,M1_c,e1,x):
        B = np.zeros(self.NS)
        B_c_tem[:-1,0], M1_c[:-1,0], _ = calc_int_BF(self.N_GRIDS * self.N_FRACS, F_norm, e1)
        v1_tem[B_c_tem != 0] = M1_c[B_c_tem != 0]/B_c_tem[B_c_tem != 0]
        v1 = v1_tem[:,0]
        B_c = B_c_tem[:,0]
        for i in range(self.NS):
            # Add contribution from LEFT cell (if existent)
            B[i] += B_c[i]*lam(v1[i], x, i, 'm')*heaviside(x[i]-v1[i],0.5)
            # Left Cell, right half
            B[i] += B_c[i-1]*lam(v1[i-1], x, i, 'm')*heaviside(v1[i-1]-x[i-1],0.5)
            # Same Cell, right half
            B[i] += B_c[i]*lam(v1[i], x, i, 'p')*heaviside(v1[i]-x[i],0.5)
            # Right Cell, left half
            B[i] += B_c[i+1]*lam(v1[i+1], x, i, 'p')*heaviside(x[i+1]-v1[i+1],0.5)
        # B /= NO_FRAG 
        prob = B.sum()
        B_vol = B * x[:-1]
        x_mass = np.sum(B * x[:-1])
        return B_vol, prob, x_mass
            
    def direct_psd_2d(self,F_norm, NO_FRAG,X1,B_c,v1,v2,M1_c,M2_c,e1,e2,x,y):
        B = np.zeros((self.NS,self.NS))
        B_c[:-1,:-1], M1_c[:-1,:-1], M2_c[:-1,:-1] = calc_int_BF(self.N_GRIDS * self.N_FRACS, F_norm[:,0], e1, F_norm[:,1], e2)
        for i in range(self.NS+1):
            for j in range(self.NS+1):
                if B_c[i,j] != 0:
                    v1[i,j] = M1_c[i,j]/B_c[i,j]
                    v2[i,j] = M2_c[i,j]/B_c[i,j]
        
        for i in range(self.NS):
            for j in range(self.NS): 
                for p in range(2):
                    for q in range(2):
                        # Actual modification calculation
                        B[i,j] += B_c[i-p,j-q] \
                            *lam_2d(v1[i-p,j-q],v2[i-p,j-q],x,y,i,j,"-","-") \
                            *heaviside((-1)**p*(x[i-p]-v1[i-p,j-q]),0.5) \
                            *heaviside((-1)**q*(y[j-q]-v2[i-p,j-q]),0.5) 
                        # B[i,j] += tem 
                        ## PRINTS FOR DEBUGGING / TESTING
                        # if i-p==0 and j-q==0:
                        #     print(f'mass flux in [{i},{j}] is', tem)
                        B[i,j] += B_c[i-p,j+q] \
                            *lam_2d(v1[i-p,j+q],v2[i-p,j+q],x,y,i,j,"-","+") \
                            *heaviside((-1)**p*(x[i-p]-v1[i-p,j+q]),0.5) \
                            *heaviside((-1)**(q+1)*(y[j+q]-v2[i-p,j+q]),0.5) 
                        # B[i,j] += tem 
                        ## PRINTS FOR DEBUGGING / TESTING
                        # if i-p==0 and j+q==0:
                        #     print(f'mass flux in [{i},{j}] is', tem)
                        B[i,j] += B_c[i+p,j-q] \
                            *lam_2d(v1[i+p,j-q],v2[i+p,j-q],x,y,i,j,"+","-") \
                            *heaviside((-1)**(p+1)*(x[i+p]-v1[i+p,j-q]),0.5) \
                            *heaviside((-1)**q*(y[j-q]-v2[i+p,j-q]),0.5)
                        # B[i,j] += tem
                        ## PRINTS FOR DEBUGGING / TESTING
                        # if i+p==0 and j-q==0:
                        #     print(f'mass flux in [{i},{j}] is', tem)
                        B[i,j] += B_c[i+p,j+q] \
                            *lam_2d(v1[i+p,j+q],v2[i+p,j+q],x,y,i,j,"+","+") \
                            *heaviside((-1)**(p+1)*(x[i+p]-v1[i+p,j+q]),0.5) \
                            *heaviside((-1)**(q+1)*(y[j+q]-v2[i+p,j+q]),0.5)
        ## Make the sum of the matrices B equal to 1, suitable for activation functions such as softmax
        # B /= NO_FRAG 
        B_vol_X1 = B * np.outer(x[:-1], np.ones(self.NS)) / X1
        B_vol_X2 = B * np.outer(np.ones(self.NS), y[:-1]) / (1-X1)
        ## Check whether the calculation results comply with mass conservation
        prob = B.sum()
        x_mass = np.sum(B_vol_X1)
        y_mass = np.sum(B_vol_X2)
        return B_vol_X1, B_vol_X2, prob, x_mass, y_mass
    
    def processing_train_data(self):
        if not os.path.exists(self.path_all_data):
            print("The processed training data is not detected and the raw data will be processed...")
            B_c,v1,v2,M1_c,M2_c,e1,e2 = self.generate_grid()
            all_Inputs, F_norm, data_num = self.load_all_data(self.directory)
            all_data, all_prob, all_x_mass, all_y_mass = self.Outputs_on_grid(all_Inputs, F_norm, data_num,
                                                                             B_c,v1,v2,M1_c,M2_c,e1,e2)
            return all_data, all_prob, all_x_mass, all_y_mass
        else:
            print(f"Processed training data detected{self.path_all_data},If you want to reprocess, please delete this file.")
        
    # %% MODEL TRAINING  
    def split_data_set(self,n_splits=5):
        self.n_splits = n_splits
        with open(self.path_all_data, 'rb') as file:
            all_data, all_prob, all_x_mass, all_y_mass = pickle.load(file)
        
        seed = None if self.global_seed == 0 else self.global_seed
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        self.train_indexs = []
        self.test_indexs = []
        if self.dim == 1:
            self.all_data = all_data[0]
            for i, (train_index, test_index) in enumerate(kf.split(self.all_data[0])):
                self.train_indexs.append(train_index)
                self.test_indexs.append(test_index)
            
        elif self.dim == 2:
            self.all_data = all_data[1]
            for i, (train_index, test_index) in enumerate(kf.split(self.all_data[0])):
                self.train_indexs.append(train_index)
                self.test_indexs.append(test_index)
            
    def create_models(self,split):
        if self.dim == 1:
            self.model = model_func.create_model_1d(self.NS, self.use_custom_loss, self.init_neurons,
                                                    self.num_layers, self.dropout_rate, self.l1_factor)
            self.model_name = f'model_1d_{split}'
            
        elif self.dim == 2:
            self.model = self.create_model_2d()
            self.model_name = f'model_2d_{split}'
            
    def create_model_2d(self):
        model_X1 = model_func.create_model_2dX1(self.NS, self.use_custom_loss, self.init_neurons,
                                                self.num_layers, self.dropout_rate, self.l1_factor)
        model_X2 = model_func.create_model_2dX2(self.NS, self.use_custom_loss, self.init_neurons,
                                                self.num_layers, self.dropout_rate, self.l1_factor)
        return model_X1, model_X2
        
    def check_x_y(self):
        if len(self.x) != self.NS:
            self.x = self.x[:-1]
        if len(self.y) != self.NS:
            self.y = self.y[:-1]
            
    def train_and_evaluate_model(self, split, epochs, training):
        self.check_x_y()
        train_index = self.train_indexs[split]
        test_index = self.test_indexs[split]
        train_data = []
        test_data = []
        for array in self.all_data:
            train_data.append(array[train_index])
            test_data.append(array[test_index])
        epochs_total = epochs * (training + 1)
        ## training and evaluation for 2d-model 
        if self.dim == 2:
            x_expand = np.outer(self.x, np.ones(self.NS))
            y_expand = np.outer(np.ones(self.NS), self.y)
            mask = np.ones((len(self.x),len(self.y)), dtype=bool)
            mask[0, 0] = False
            # train model
            if self.use_custom_loss:
                results = model_func.train_model_2d(self.model, train_data, test_data, x_expand, y_expand, mask, self.print_status,
                                          self.weight_loss_FRAG_NUM, epochs, self.batch_size,
                                          self.optimizer_type, self.learning_rate)
            else:
                self.model[0].fit(train_data[3], train_data[1], epochs=epochs, batch_size=32, validation_split=0.2)   
                self.model[1].fit(train_data[3], train_data[2], epochs=epochs, batch_size=32, validation_split=0.2)
            if self.save_model:
                self.model[0].save(self.model_name + f'X1_{epochs_total}.keras')
                self.model[1].save(self.model_name + f'X2_{epochs_total}.keras')

            # test_res = []
        ## training and evaluation for 2d-model 
        if self.dim == 1:
            mask = np.ones(self.x.shape, dtype=bool)
            mask[0] = False
            # train model
            if self.use_custom_loss:
                results = model_func.train_model_1d(self.model, train_data, test_data, self.x, mask, self.print_status, self.weight_loss_FRAG_NUM,
                                          epochs, self.batch_size, self.optimizer_type, self.learning_rate)
            else:
                self.model.fit(train_data[2], train_data[1], epochs=epochs, batch_size=32, validation_split=0.2)   
            if self.save_model:
                self.model.save(self.model_name + f'_{epochs_total}.keras')
                
        return results
    
    def cross_validation(self, epochs, num_training):
        ## 2rd dimension of results is: mse, mae, number_error, x_mass_error, y_mass_error
        results = np.zeros((num_training,5))
        for split in range(self.n_splits):
            self.create_models(split)
            for training in range(num_training):
                results[training] += self.train_and_evaluate_model(split, epochs, training)
        results /= self.n_splits
        if self.save_validate_results:
            with open(f'epochs_{epochs*num_training}_weight_{self.weight_loss_FRAG_NUM}.pkl', 'wb') as f:
                pickle.dump(results, f)
        return results
    
    # %% POST PROCESSING  
    def precalc_matrix_ANN_to_B_F(self, NS, NSS, V1, V3, V_e1, V_e3,x,y):
        precalc_matrix = np.zeros((NSS, NSS, NSS, NSS, NS, NS))
        
        for k in range(NSS):
            for l in range(NSS):
                if k+l == 0:
                    continue
                ## k, l represent the volume of the original particle
                V1_kl = V1[k]
                V3_kl = V3[l]
                ## Volume distribution of fragments generated by current particles
                fragment_V1 = V1_kl * x[:, np.newaxis]
                fragment_V3 = V3_kl * y[np.newaxis, :]
                
                for m in range(k + 1):
                    for n in range(l + 1):
                        mask1 = (V_e1[m] <= fragment_V1) & (fragment_V1 < V_e1[m + 1])
                        mask2 = (V_e3[n] <= fragment_V3) & (fragment_V3 < V_e3[n + 1])
                        ## The sum of the probability of fragments falling in each cell
                        precalc_matrix[m, n, k, l] = mask1 & mask2
    
        return precalc_matrix
    
    def calc_B_F(self, FRAG, precalc_matrix, NSS):
        # B_F = np.einsum('klmnij,ijkl->klmn', precalc_matrix, FRAG)
        int_B_F = np.zeros((NSS,NSS,NSS,NSS))
        intx_B_F = np.zeros((NSS,NSS,NSS,NSS))
        inty_B_F = np.zeros((NSS,NSS,NSS,NSS))
        for k in range(NSS):
                for l in range(NSS):
                    for m in range(k + 1):
                        for n in range(l + 1):
                            int_B_F[m, n, k, l] = np.sum(precalc_matrix[m, n, k, l] * FRAG[0,k,l])
                            intx_B_F[m, n, k, l] = np.sum(precalc_matrix[m, n, k, l] * FRAG[1,k,l])
                            inty_B_F[m, n, k, l] = np.sum(precalc_matrix[m, n, k, l] * FRAG[2,k,l])
            
        return int_B_F, intx_B_F, inty_B_F
    def calc_FRAG(self, epochs_total, V, V_rel, X1, STR1, STR2, STR3, NO_FRAG, int_bre, NS,NSS, x,y, mask):
        ## FRAG stores the information of NS*NS fragments broken into by NSS*NSS particles. 
        ## The first dimension stores the number of fragments. 
        ## The second dimension is the X1 volume of the fragment. 
        ## The third dimension is the volume of X2
        FRAG = np.zeros((3,NSS,NSS,NS,NS))
        model_1d = load_model(f'model_1d_{epochs_total}.keras')
        model_2dX1 = load_model(f'model_2dX1_{epochs_total}.keras')
        model_2dX2 = load_model(f'model_2dX2_{epochs_total}.keras')
        with open(self.path_scaler,'rb') as file:
            scaler_1d, scaler_2d = pickle.load(file)
        ## calculate 1d FRAG
        V_rel_1d = V_rel[1:,0]
        num_elements = V_rel_1d.size
        NO_FRAG_array = np.full(num_elements, NO_FRAG)
        int_bre_array = np.full(num_elements, int_bre)
        inputs_1d = np.column_stack((V_rel_1d, NO_FRAG_array, int_bre_array))
        inputs_1d_scaled = scaler_1d.fit_transform(inputs_1d)
        FRAG_vol_1d = model_1d.predict(inputs_1d_scaled)
        for i in range(1,NSS):
            FRAG[0, i,0,1:,0] = FRAG_vol_1d[i-1,mask[0]] / x[mask[0]]
            FRAG[1, i,0,1:,0] = FRAG_vol_1d[i-1,mask[0]] * V[i,0]
            FRAG[2, i,0,1:,0] = 0.0
            
        V_rel_1d = V_rel[0,1:]
        num_elements = V_rel_1d.size
        NO_FRAG_array = np.full(num_elements, NO_FRAG)
        int_bre_array = np.full(num_elements, int_bre)
        inputs_1d = np.column_stack((V_rel_1d, NO_FRAG_array, int_bre_array))
        inputs_1d_scaled = scaler_1d.fit_transform(inputs_1d)
        FRAG_vol_1d = model_1d.predict(inputs_1d)
        for i in range(1,NSS):
            FRAG[0, 0,i,0,1:] = FRAG_vol_1d[i-1,mask[0]] / y[mask[0]]
            FRAG[1, 0,i,0,1:] = 0.0
            FRAG[2, 0,i,0,1:] = FRAG_vol_1d[i-1,mask[0]] * V[0,i]
            
        V_rel_2d_flat = V_rel[1:,1:].flatten()
        X1_2d_flat = X1[1:,1:].flatten()
        num_elements = V_rel_2d_flat.size
        STR1_array = np.full(num_elements, STR1)
        STR2_array = np.full(num_elements, STR2)
        STR3_array = np.full(num_elements, STR3)
        NO_FRAG_array = np.full(num_elements, NO_FRAG)
        int_bre_array = np.full(num_elements, int_bre)
        inputs_2d = np.column_stack((V_rel_2d_flat, X1_2d_flat, STR1_array, STR2_array, STR3_array, NO_FRAG_array, int_bre_array))
        inputs_2d_scaled = scaler_2d.fit_transform(inputs_2d)
        FRAG_vol_2dX1 = model_2dX1.predict(inputs_2d_scaled).reshape(NSS-1, NSS-1, NS, NS)
        FRAG_vol_2dX2 = model_2dX2.predict(inputs_2d_scaled).reshape(NSS-1, NSS-1, NS, NS)
        x_expand = np.outer(x, np.ones(NS))
        y_expand = np.outer(np.ones(NS), y)
        for i in range(1,NSS):
            for j in range(1,NSS):
                V_inv = x_expand*X1[i,j] + y_expand*(1-X1[i,j])
                FRAG[0, i,j,mask] = (FRAG_vol_2dX1[i-1,j-1,mask]+FRAG_vol_2dX2[i-1,j-1,mask]) / V_inv[mask]
                FRAG[1, i,j,mask] = FRAG_vol_2dX1[i-1,j-1,mask] * X1[i,j] * V[i,j]
                FRAG[2, i,j,mask] = FRAG_vol_2dX2[i-1,j-1,mask] * (1-X1[i,j]) * V[i,j]
        return FRAG
        
    def test_ANN_to_B_F(self, NS,S,epochs_total):
        STR1 = 0.5
        STR2 = 0.5
        STR3 = 0.5
        NO_FRAG = 4
        int_bre = 0.5
        NSS = 15
        SS = 2
        V1 = np.zeros(NSS)
        V3 = np.copy(V1)
        V_e1 = np.zeros(NSS+1)
        V_e3 = np.copy(V_e1)
        V_e1[0] = -math.pi * 2
        V_e3[0] = -math.pi * 2
        for i in range(NSS):
            V_e1[i+1] = SS ** (i) * math.pi * 2
            V_e3[i+1] = SS ** (i) * math.pi * 2
            V1[i] = (V_e1[i]+V_e1[i+1]) / 2
            V3[i] = (V_e3[i]+V_e3[i+1]) / 2
        V = np.zeros((NSS,NSS))
        X1 = np.copy(V)
        for i in range(NSS):
            for j in range(NSS):
                V[i,j] =  V1[i] + V3[j]
                if i==0 and j==0:
                    X1[i,j] = 0
                else:
                    X1[i,j] = V1[i] / V[i,j]
        ## The volume V00 of the main particles of the system, 
        ## this volume directly corresponds to the model input parameters. 
        ## The volume of other particles is defined relative to V00
        V00 = 2
        V_rel = V / min(V1[1],V3[1]) * V00
        
        e1 = np.zeros(NS+1)
        e2 = np.zeros(NS+1)
        x = np.zeros(NS)
        y = np.zeros(NS)
        e1[0] = -1
        e2[0] = -1
        for i in range(NS):
            e1[i+1] = S**(i)
            e2[i+1] = S**(i)
            x[i] = (e1[i] + e1[i+1]) / 2
            y[i] = (e2[i] + e2[i+1]) / 2
        V_xy = np.zeros((len(x),len(y)))
        X1_xy = np.zeros_like(V_xy)
        for i, v_x in enumerate(x):
            for j, v_y in enumerate(y):
                if i + j != 0:
                    V_xy[i,j] = v_x + v_y
                    X1_xy[i,j] = v_x / V_xy[i,j]
        x /= x[-1]
        y /= y[-1]
    
        mask = np.ones(V_xy.shape, dtype=bool)
        mask[0, 0] = False
        
        FRAG = self.calc_FRAG(epochs_total, V, V_rel, X1, STR1, STR2, STR3, NO_FRAG, int_bre, NS, NSS, x,y, mask)
        precalc_matrix = self.precalc_matrix_ANN_to_B_F(NS,NSS,V1,V3,V_e1,V_e3,x,y)
        int_B_F, intx_B_F, inty_B_F = self.calc_B_F(FRAG,precalc_matrix,NSS)
        
        np.savez('int_B_F.npz',
                 int_B_F=int_B_F,
                 intx_B_F = intx_B_F,
                 inty_B_F = inty_B_F)
        
        return int_B_F, intx_B_F, inty_B_F, V, FRAG
    
    def plot_error(self, epochs,num_training, results=None):
        if results is None:
            try: 
                with open(f'epochs_{epochs*num_training}_weight_{self.weight_loss_FRAG_NUM}.pkl', 'rb') as f:
                    results = pickle.load(f)
            except FileNotFoundError:
                print('please use cross_validation() to generate the results file first!')
                return
                
        epochs_array = np.arange(epochs, epochs*num_training+1, epochs)
        pt.close()
        pt.plot_init(scl_a4=1,figsze=[12.8,6.4*1.5],lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
        fig1=plt.figure()
        fig2=plt.figure()
        fig3=plt.figure()
        fig4=plt.figure()
        ax1=fig1.add_subplot(1,1,1)
        ax2=fig2.add_subplot(1,1,1)
        ax3=fig3.add_subplot(1,1,1) 
        ax4=fig4.add_subplot(1,1,1)
        
        ax1, fig1 = pt.plot_data(epochs_array, results[:,0],fig=fig1,ax=ax1,
                               xlbl='Epochs of Model Training / $-$',
                               ylbl='Results of Cross Validation / $-$',
                               lbl=f'mse_{self.dim}d',clr='b',mrk='o')
        
        ax2, fig2 = pt.plot_data(epochs_array, results[:,1],fig=fig2,ax=ax2,
                               xlbl='Epochs of Model Training / $-$',
                               ylbl='Results of Cross Validation / $-$',
                               lbl=f'mae_{self.dim}d',clr='b',mrk='o')
        
        ax3, fig3 = pt.plot_data(epochs_array, results[:,2],fig=fig3,ax=ax3,
                               xlbl='Epochs of Model Training / $-$',
                               ylbl='Results of Cross Validation / $-$',
                               lbl=f'mean_frag_erro_{self.dim}d',clr='b',mrk='o')
    
        ax4, fig4 = pt.plot_data(epochs_array, results[:,3],fig=fig4,ax=ax4,
                               xlbl='Epochs of Model Training / $-$',
                               ylbl='Results of Cross Validation / $-$',
                               lbl=f'mean_x_mass_erro_{self.dim}d',clr='b',mrk='o')
        if self.dim != 1:
            ax4, fig4 = pt.plot_data(epochs_array,results[:,4],fig=fig4,ax=ax4,
                                   lbl=f'mean_y_mass_erro_{self.dim}d',clr='r',mrk='^')
        
        return   
    
    def plot_1d_F(self, x,NS,test_data,epochs_total,data_index=0,vol_dis=False):
        model = load_model(f'model_1d_{epochs_total}.keras')
        test_Input = test_data[2][data_index].reshape(1,3)
        test_Output = test_data[1][data_index]
        predicted_Output = model.predict(test_Input).reshape(NS)
        ylbl = 'Total Relative Volume / $-$'
        if not vol_dis:
            mask = np.ones(x.shape, dtype=bool)
            mask[0] = False
            test_Output[mask] /= x[mask]
            test_Output[0] = 0.0
            predicted_Output[mask] /= x[mask]
            test_Output[0] = 0.0
            ylbl = 'Counts / $-$'
        ## Convert to quantity distribution matrix counts
        test_counts = (1e5 * test_Output).astype(int)
        predicted_counts = (1e5 * predicted_Output).astype(int)
        x_test_counts = []
        x_predicted_counts = []
        for i in range(test_counts.shape[0]):
            x_test_counts.extend([x[i]] * test_counts[i])
            x_predicted_counts.extend([x[i]] * predicted_counts[i])
        x_test_counts = np.array(x_test_counts)
        x_predicted_counts = np.array(x_predicted_counts)
        
        ax1, fig1, H1, xe1 = pt.plot_1d_hist(x=x_test_counts,bins=100,scale='line',xlbl='Relative Volume of Fragments / $-$',
                                             ylbl=ylbl,tit='LMC',clr=c_KIT_green,norm=False, alpha=0.7)
        ax2, fig2, H2, xe2 = pt.plot_1d_hist(x=x_predicted_counts,bins=100,scale='line',xlbl='Relative Volume of Fragments / $-$',
                                             ylbl=ylbl,tit='ANN',clr=c_KIT_green,norm=False, alpha=0.7)
        return ax1, ax2
        
    def plot_2d_F(self, x,y,NS,test_data,epochs_total,data_index=0,vol_dis=False):
        model_X1 = load_model(f'model_2dX1_{epochs_total}.keras')
        model_X2 = load_model(f'model_2dX2_{epochs_total}.keras')
        test_Input = test_data[3][data_index].reshape(1,7)
        test_OutputX1 = test_data[1][data_index]
        test_OutputX2 = test_data[2][data_index]
        predicted_OutputX1 = model_X1.predict(test_Input).reshape(NS,NS)
        predicted_OutputX2 = model_X2.predict(test_Input).reshape(NS,NS)
        test_Output = test_OutputX1+test_OutputX2
        predicted_Output = predicted_OutputX1+predicted_OutputX2
        if not vol_dis:
            mask = np.ones((len(x),len(y)), dtype=bool)
            V = np.zeros((len(x),len(y)))
            mask[0, 0] = False
            X1 = test_Input[0,1]
            X2 = 1 - X1
            for i in range(len(x)):
                for j in range(len(y)):
                    V[i,j] = x[i] / X1 + y[j] / X2
            test_Output[mask] = (test_OutputX1[mask]+test_OutputX2[mask])/ V[mask]     
            predicted_Output[mask] = (predicted_OutputX1[mask]+predicted_OutputX2[mask])/ V[mask]
        test_counts = (1e5 * test_Output).astype(int)
        predicted_counts = (1e5 * predicted_Output).astype(int)
        x_test_counts = []
        x_predicted_counts = []
        y_test_counts = []
        y_predicted_counts = []
        for i in range(test_counts.shape[0]):
            for j in range(test_counts.shape[1]):
                x_test_counts.extend([x[i]] * test_counts[i,j])
                x_predicted_counts.extend([x[i]] * predicted_counts[i,j])
                y_test_counts.extend([y[j]] * test_counts[i,j])
                y_predicted_counts.extend([y[j]] * predicted_counts[i,j])
        x_test_counts = np.array(x_test_counts)
        x_predicted_counts = np.array(x_predicted_counts)
        y_test_counts = np.array(y_test_counts)
        y_predicted_counts = np.array(y_predicted_counts)
        ax1, fig1, cb1, H1, xe1, ye1 = pt.plot_2d_hist(x=x_test_counts,y=y_test_counts,bins=(50,50),w=None,
                                                       scale=('line','line'), clr=KIT_black_green_white.reversed(), 
                                                       xlbl='Relative Volume V1 of Fragments / $-$', norm=False,
                                                       ylbl='Relative Volume V2 of Fragments / $-$', grd=True,
                                                       scale_hist='log', hist_thr=1e-4,tit='LMC')
        ax2, fig2, cb2, H2, xe2, ye2 = pt.plot_2d_hist(x=x_predicted_counts,y=y_predicted_counts,bins=(50,50),w=None,
                                                       scale=('line','line'), clr=KIT_black_green_white.reversed(), 
                                                       xlbl='Relative Volume V1 of Fragments / $-$', norm=False,
                                                       ylbl='Relative Volume V2 of Fragments / $-$', grd=True,
                                                       scale_hist='log', hist_thr=1e-4,tit='ANN')
        return ax1, ax2
    

# %% MAIN
if __name__ == '__main__':
    ## The value of random seed (int value) itself is not important.
    ## But fixing random seeds can ensure the consistency and comparability of the results.
    ## The reverse improves the robustness of the model (set m_global_seed=0)
    m_global_seed = 0
    
    m_dim = 2
    m_NS = 50
    m_S = 1.3
    m_n_splits = 5
    m_epochs = 1
    m_num_training = 1
    
    ann = ANN_bond_break(m_dim,m_NS,m_S,m_global_seed)
    ann.save_model = False
    ann.save_validate_results = False
    ann.processing_train_data()
    ann.split_data_set(n_splits=m_n_splits)
    results = ann.cross_validation(m_epochs, m_num_training)
    ann.plot_error(m_epochs, m_num_training,results)
    