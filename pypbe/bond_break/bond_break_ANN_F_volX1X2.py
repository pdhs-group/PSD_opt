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
## external package
import math
from keras import regularizers
from keras.models import Sequential, load_model
from keras.layers import Dense, InputLayer, Reshape, Lambda, BatchNormalization
from keras.optimizers import Adam
from keras.saving import register_keras_serializable
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler
## for plotter
import matplotlib.pyplot as plt
import pypbe.utils.plotter.plotter as pt
from pypbe.utils.plotter.KIT_cmap import c_KIT_green, c_KIT_red, c_KIT_blue, KIT_black_green_white

def load_all_data(directory):
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
def generate_grid():
    e1 = np.zeros(NS+1)
    e2 = np.zeros(NS+1)
    ## Because of the subsequent CAT processing, x and y need to be expanded 
    ## outward by one unit with value -1
    x = np.zeros(NS+1)-1
    y = np.zeros(NS+1)-1
    e1[0] = -1
    e2[0] = -1
    for i in range(NS):
        e1[i+1] = S**(i)
        e2[i+1] = S**(i)
        x[i] = (e1[i] + e1[i+1]) / 2
        y[i] = (e2[i] + e2[i+1]) / 2
    e1[:] /= x[-2] 
    e2[:] /= y[-2] 
    x[:] /= x[-2] 
    y[:] /= y[-2] 
    B_c = np.zeros((NS+1,NS+1)) 
    v1 = np.zeros((NS+1,NS+1))
    v2 = np.zeros((NS+1,NS+1))
    M1_c = np.zeros((NS+1,NS+1))
    M2_c = np.zeros((NS+1,NS+1))
    return B_c,v1,v2,M1_c,M2_c,e1,e2,x,y

def Outputs_on_grid(path_all_data,path_scaler, all_Inputs, F_norm, data_num,
                    B_c,v1,v2,M1_c,M2_c,e1,e2,x,y,test_data):
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
            tem_Outputs, all_prob[i], all_x_mass[i] = direct_psd_1d(F_norm_tem[:,0],NO_FRAG,B_c,v1,M1_c,e1,x)
            Outputs_1d.append(tem_Outputs)
            tem_Inputs = [all_Inputs[i,j] for j in range(7) if j == 0 or j == 2 or j == 6]
            Inputs_1d.append(tem_Inputs)
            mean_energy = F_norm_tem[:,2].mean()
            Outputs_1d_E.append(mean_energy)
        else:
            tem_OutputsX1,tem_OutputsX2, all_prob[i], all_x_mass[i], all_y_mass[i]= direct_psd_2d(F_norm_tem,NO_FRAG,X1,B_c,v1,v2,M1_c,M2_c,e1,e2,x,y)   
            Outputs_2dX1.append(tem_OutputsX1)
            Outputs_2dX2.append(tem_OutputsX2)
            Inputs_2d.append(all_Inputs[i,:])
            mean_energy = F_norm_tem[:,2].mean()
            Outputs_2d_E.append(mean_energy)
    if not test_data:
        scaler_1d = StandardScaler()
        scaler_2d = StandardScaler()
        with open(path_scaler,'wb') as file:
            pickle.dump((scaler_1d,scaler_2d), file)
    else:
        with open(path_scaler,'rb') as file:
            scaler_1d, scaler_2d = pickle.load(file)
    Inputs_1d_scaled = scaler_1d.fit_transform(Inputs_1d)
    Inputs_2d_scaled = scaler_2d.fit_transform(Inputs_2d)
    all_data = [[np.array(Inputs_1d),np.array(Outputs_1d),np.array(Inputs_1d_scaled),np.array(Outputs_1d_E)],
                [np.array(Inputs_2d),np.array(Outputs_2dX1),np.array(Outputs_2dX2),np.array(Inputs_2d_scaled),np.array(Outputs_2d_E)]]

    with open(path_all_data, 'wb') as file:
        pickle.dump((all_data, all_prob, all_x_mass, all_y_mass), file)    
        
    return all_data, all_prob, all_x_mass, all_y_mass

def direct_psd_1d(F_norm,NO_FRAG,B_c_tem,v1_tem,M1_c,e1,x):
    B = np.zeros(NS)
    B_c_tem[:-1,0], M1_c[:-1,0], _ = calc_int_BF(N_GRIDS * N_FRACS, F_norm, e1)
    v1_tem[B_c_tem != 0] = M1_c[B_c_tem != 0]/B_c_tem[B_c_tem != 0]
    v1 = v1_tem[:,0]
    B_c = B_c_tem[:,0]
    for i in range(NS):
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
        
def direct_psd_2d(F_norm, NO_FRAG,X1,B_c,v1,v2,M1_c,M2_c,e1,e2,x,y):
    B = np.zeros((NS,NS))
    B_c[:-1,:-1], M1_c[:-1,:-1], M2_c[:-1,:-1] = calc_int_BF(N_GRIDS * N_FRACS, F_norm[:,0], e1, F_norm[:,1], e2)
    for i in range(NS+1):
        for j in range(NS+1):
            if B_c[i,j] != 0:
                v1[i,j] = M1_c[i,j]/B_c[i,j]
                v2[i,j] = M2_c[i,j]/B_c[i,j]
    
    for i in range(NS):
        for j in range(NS): 
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
    B_vol_X1 = B * np.outer(x[:-1], np.ones(NS)) / X1
    B_vol_X2 = B * np.outer(np.ones(NS), y[:-1]) / (1-X1)
    ## Check whether the calculation results comply with mass conservation
    prob = B.sum()
    x_mass = np.sum(B_vol_X1)
    y_mass = np.sum(B_vol_X2)
    return B_vol_X1, B_vol_X2, prob, x_mass, y_mass
# %% MODEL TRAINING  
@register_keras_serializable()
def pad_Outputs_X1(Outputs):
    batch_size = tf.shape(Outputs)[0]
    padding = tf.zeros((batch_size, 1, NS), dtype=Outputs.dtype)
    padde_Outputs = tf.concat([padding, Outputs], axis=1)
    return padde_Outputs
@register_keras_serializable()
def pad_Outputs_X2(Outputs):
    batch_size = tf.shape(Outputs)[0]
    padding = tf.zeros((batch_size, NS, 1), dtype=Outputs.dtype)
    padde_Outputs = tf.concat([padding, Outputs], axis=2)
    return padde_Outputs
def create_model_2dX1X2():
    model_X1 = create_model_2dX1()
    model_X2 = create_model_2dX2()
    return model_X1, model_X2
def create_model_2dX1():
    # The input layer accepts an array of length 7 (combined STR and FRAG)
    # The output layer should now match the flattened shape of the new combined output array
    model = Sequential([
        InputLayer(shape=(7,)),  
        Dense(256, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
        BatchNormalization(),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
        BatchNormalization(),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
        BatchNormalization(),
        Dense((NS-1)*NS, activation='softmax'),  # Adjusted for the new output shape
        Reshape((NS-1, NS)),  # Reshape the output to the desired shape
        Lambda(pad_Outputs_X1) # Add zero rows in front of first dimension
    ])
    if not use_custom_loss:
        model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    return model

def create_model_2dX2():
    # The input layer accepts an array of length 7 (combined STR and FRAG)
    # The output layer should now match the flattened shape of the new combined output array
    model = Sequential([
        InputLayer(shape=(7,)),  
        Dense(256, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
        BatchNormalization(),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
        BatchNormalization(),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
        BatchNormalization(),
        Dense(NS*(NS-1), activation='softmax'),  # Adjusted for the new output shape
        Reshape((NS, NS-1)),  # Reshape the output to the desired shape
        Lambda(pad_Outputs_X2) # Add zero column in front of second dimension
    ])
    if not use_custom_loss:
        model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    return model

def create_model_1d():
    # The input layer accepts an array of length 4 (combined STR and FRAG)
    # The output layer should now match the flattened shape of the new combined output array
    model = Sequential([
        InputLayer(shape=(3,)),  
        Dense(128, activation='relu', kernel_regularizer=regularizers.l1(0.001)),
        BatchNormalization(),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.001)),
        BatchNormalization(),
        Dense(NS, activation='softmax'),  # Adjusted for the new output shape
    ])
    if not use_custom_loss:
        model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    return model

@tf.function
def compute_loss_and_grads(model, inputs, outputs, V_tf, other_pred, mask_tf, FRAG_NUM, weight_loss_FRAG_NUM):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = combined_custom_loss(outputs, predictions, FRAG_NUM, V_tf, mask_tf, other_pred, alpha=weight_loss_FRAG_NUM, beta=1-weight_loss_FRAG_NUM)
    grads = tape.gradient(loss, model.trainable_weights)
    return loss, grads

def train_model_2d(model, train_data, x, y, mask, epochs=100, batch_size=32):
    model_X1 = model[0]
    model_X2 = model[1]
    all_Inputs = train_data[0]
    all_Outputs_X1 = train_data[1]
    all_Outputs_X2 = train_data[2]
    all_Inputs_scaled = train_data[3]
    
    X1 = all_Inputs[:,1]
    X2 = 1 - X1
    FRAG_NUM = all_Inputs[:, 2]
    V = x / X1[:, np.newaxis, np.newaxis] + y / X2[:, np.newaxis, np.newaxis]
    
    # optimizer=Adam(learning_rate=0.0001)
    optimizer_X1=Adam()
    optimizer_X2=Adam()
    train_dataset = tf.data.Dataset.from_tensor_slices((
        all_Inputs.astype(np.float32),
        all_Outputs_X1.astype(np.float32),
        all_Outputs_X2.astype(np.float32),
        all_Inputs_scaled.astype(np.float32),
        V.astype(np.float32),
        FRAG_NUM.astype(np.float32)
    )).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    mask_tf = tf.convert_to_tensor(mask)
    for epoch in range(epochs):
        print(f'Starting epoch {epoch+1}')

        for step, (batch_inputs, batch_outputs_X1, batch_outputs_X2, 
                   batch_inputs_scaled, batch_V_tf, batch_FRAG_NUM) in enumerate(train_dataset):

            loss_X1, grads_X1 = compute_loss_and_grads(model_X1, batch_inputs_scaled, batch_outputs_X1, 
                                                       batch_V_tf, model_X2(batch_inputs_scaled, training=False), 
                                                       mask_tf, batch_FRAG_NUM, weight_loss_FRAG_NUM)
            
            loss_X2, grads_X2 = compute_loss_and_grads(model_X2, batch_inputs_scaled, batch_outputs_X2, 
                                                       batch_V_tf, model_X1(batch_inputs_scaled, training=False),
                                                       mask_tf, batch_FRAG_NUM, weight_loss_FRAG_NUM)
            
            optimizer_X1.apply_gradients(zip(grads_X1, model_X1.trainable_weights))
            optimizer_X2.apply_gradients(zip(grads_X2, model_X2.trainable_weights))
            
            print(f'Training step {step+1}, Loss_X1: {loss_X1.numpy().mean()}, Loss_X2: {loss_X2.numpy().mean()}')

    print(f"Training of Echos = {epochs} completed!")
    
def train_model_1d(model, train_data, x, mask, epochs=100, batch_size=32):
    all_Inputs = train_data[0]
    all_Outputs = train_data[1]
    all_Inputs_scaled = train_data[2]
    # optimizer=Adam(learning_rate=0.0001)
    optimizer=Adam()
    train_dataset = tf.data.Dataset.from_tensor_slices((all_Inputs.astype(np.float32), all_Outputs.astype(np.float32), 
                                                        all_Inputs_scaled.astype(np.float32))).batch(batch_size)
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    mask_tf = tf.convert_to_tensor(mask)
    for epoch in range(epochs):
        print(f'Starting epoch {epoch+1}')
        # Training
        for step, (batch_inputs, batch_outputs, batch_inputs_scaled) in enumerate(train_dataset):
            FRAG_NUM = batch_inputs[:,1]
            with tf.GradientTape() as tape:
                prediction  = model(batch_inputs_scaled, training=True)
                loss = combined_custom_loss(batch_outputs, prediction, FRAG_NUM, x_tf, mask_tf, 
                                            alpha=weight_loss_FRAG_NUM, beta=1-weight_loss_FRAG_NUM)
            
            grads = tape.gradient(loss, model.trainable_weights)
            # grads = [tf.clip_by_value(g, -1.0, 1.0) for g in grads]
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            print(f'Training step {step+1}, Loss: {loss.numpy().mean()}')
        
    print(f"Training of Echos = {epochs} completed!")
def custom_loss_all_prob(FRAG_NUM, V, mask, y_pred_other):
    def loss(y_true, y_pred): 
        if y_pred_other is None:
            pre_all_prob = tf.reduce_sum(tf.where(mask, y_pred / (V+1e-20), 0), axis=1)
        else:
            pre_all_prob = tf.reduce_sum(tf.where(mask, (y_pred+y_pred_other) / (V+1e-20), 0), axis=[1, 2])
        
        relative_error_all_prob = abs((pre_all_prob - FRAG_NUM)) / FRAG_NUM
        mse_all_prob = tf.reduce_mean(tf.square(relative_error_all_prob))

        return mse_all_prob

    return loss

def combined_custom_loss(y_true, y_pred, FRAG_NUM, V, mask, y_pred_other=None, alpha=0.5, beta=0.5):
    custom_loss_func_all_prob = custom_loss_all_prob(FRAG_NUM, V, mask, y_pred_other)
    loss_custom_all_prob = custom_loss_func_all_prob(y_true, y_pred)
    # loss_mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    loss_kl = tf.keras.losses.kl_divergence(y_true, y_pred)

    return alpha * loss_custom_all_prob + beta * loss_kl

def train_and_evaluate_model(model, model_name, train_data, test_data, epochs, training, x, y=None):
    epochs_total = epochs * (training + 1)
    ## training and evaluation for 2d-model 
    if y is not None:
        x_expand = np.outer(x, np.ones(NS))
        y_expand = np.outer(np.ones(NS), y)
        mask = np.ones((len(x),len(y)), dtype=bool)
        mask[0, 0] = False
        # train model
        if use_custom_loss:
            train_model_2d(model, train_data, epochs=epochs, x=x_expand, y=y_expand, mask=mask)
        else:
            model[0].fit(train_data[3], train_data[1], epochs=epochs, batch_size=32, validation_split=0.2)   
            model[1].fit(train_data[3], train_data[2], epochs=epochs, batch_size=32, validation_split=0.2)
      
        model[0].save(model_name + f'X1_{epochs_total}.keras')
        model[1].save(model_name + f'X2_{epochs_total}.keras')
        # evaluate_model
        test_res = predictions_test_2d(model_name, test_data, epochs_total,x=x_expand, y=y_expand, mask=mask)
        # test_res = []
    ## training and evaluation for 2d-model 
    else:
        mask = np.ones(x.shape, dtype=bool)
        mask[0] = False
        # train model
        if use_custom_loss:
            train_model_1d(model, train_data, epochs=epochs, x=x, mask=mask)
        else:
            model.fit(train_data[2], train_data[1], epochs=epochs, batch_size=32, validation_split=0.2)   
      
        model.save(model_name + f'_{epochs_total}.keras')
        # evaluate_model
        test_res = predictions_test_1d(model_name, test_data, epochs_total,x=x, mask=mask)
    return test_res

def predictions_test_1d(model_name,test_data, epochs_total,x, mask): 
    test_Inputs = test_data[0]
    test_Outputs = test_data[1]
    test_Inputs_scaled = test_data[2]
    model = load_model(model_name + f'_{epochs_total}.keras')
    predicted_Outputs = model.predict(test_Inputs_scaled)
    if use_custom_loss:
        all_mse = tf.keras.losses.mean_squared_error(test_Outputs, predicted_Outputs).numpy()
        all_mae = tf.keras.losses.mean_absolute_error(test_Outputs, predicted_Outputs).numpy()
        mse = all_mse.mean()
        mae = all_mae.mean()
    else:
        mse, mae = model.evaluate(test_Inputs_scaled, test_Outputs)
    length = test_Inputs.shape[0]
    pre_all_prob = np.zeros(length)
    pre_all_x_mass = np.zeros(length)
    ## In the case of 1d, test_Inputs[i,1] represents the number of fragments
    ## In the case of 1d, test_Inputs[i,1] represents the Volume fraction of X1,
    ## test_Inputs[i,2] represents the number of fragments
    true_all_prob =  test_Inputs[:,1]
    true_all_x_mass = np.ones_like(true_all_prob)

    for i in range(length):
        pre_all_prob[i] = np.sum(predicted_Outputs[i,mask] / x[mask])
        pre_all_x_mass[i] = np.sum(predicted_Outputs[i,:])
            
    mean_all_prob = (abs(true_all_prob - pre_all_prob) / true_all_prob).mean()
    mean_all_x_mass = (abs(true_all_x_mass - pre_all_x_mass) / true_all_x_mass).mean()
    mean_all_y_mass = 1

    return mse, mae, mean_all_prob, mean_all_x_mass, mean_all_y_mass

def predictions_test_2d(model_name,test_data,epochs_total,x,y,mask): 
    test_Inputs = test_data[0]
    test_Outputs_X1 = test_data[1]
    test_Outputs_X2 = test_data[2]
    test_Inputs_scaled = test_data[3]
    model_X1 = load_model(model_name+ f'X1_{epochs_total}.keras')
    model_X2 = load_model(model_name+ f'X2_{epochs_total}.keras')
    predicted_Outputs_X1 = model_X1.predict(test_Inputs_scaled)
    predicted_Outputs_X2 = model_X2.predict(test_Inputs_scaled)
    if use_custom_loss:
        all_mse_X1 = tf.keras.losses.mean_squared_error(test_Outputs_X1, predicted_Outputs_X1).numpy()
        all_mae_X1 = tf.keras.losses.mean_absolute_error(test_Outputs_X1, predicted_Outputs_X1).numpy()
        all_mse_X2 = tf.keras.losses.mean_squared_error(test_Outputs_X2, predicted_Outputs_X2).numpy()
        all_mae_X2 = tf.keras.losses.mean_absolute_error(test_Outputs_X2, predicted_Outputs_X2).numpy()
        mse = (all_mse_X1+all_mse_X2).mean()
        mae = (all_mae_X1+all_mae_X2).mean()
    else:
        mse_X1, mae_X1 = model_X1.evaluate(test_Inputs_scaled, test_Outputs_X1)
        mse_X2, mae_X2 = model_X2.evaluate(test_Inputs_scaled, test_Outputs_X2)
        mse = (mse_X1+mse_X2)/2
        mae = (mae_X1+mae_X2)/2
    length = test_Inputs.shape[0]
    pre_all_prob = np.zeros(length)
    pre_all_x_mass = np.zeros(length)
    pre_all_y_mass = np.zeros(length)
    ## In the case of 1d, test_Inputs[i,1] represents the number of fragments
    ## In the case of 1d, test_Inputs[i,1] represents the Volume fraction of X1,
    ## test_Inputs[i,2] represents the number of fragments
    X1 = test_Inputs[:,1]
    X2 = 1 - X1
    true_all_prob =  test_Inputs[:,2]
    true_all_x_mass = np.ones_like(true_all_prob)
    true_all_y_mass = np.ones_like(true_all_prob)
    for i in range(length):
        V = x / X1[i] + y /X2[i]
        pre_all_prob[i] = np.sum((predicted_Outputs_X1[i,mask]+predicted_Outputs_X2[i,mask])/ V[mask])
        pre_all_x_mass[i] = np.sum(predicted_Outputs_X1[i,:])
        pre_all_y_mass[i] = np.sum(predicted_Outputs_X2[i,:])
            
    mean_all_prob = (abs(true_all_prob - pre_all_prob) / true_all_prob).mean()
    mean_all_x_mass = (abs(true_all_x_mass - pre_all_x_mass)).mean()
    mean_all_y_mass = (abs(true_all_y_mass - pre_all_y_mass)).mean()

    return mse, mae, mean_all_prob, mean_all_x_mass, mean_all_y_mass

# %% POST PROCESSING  
def precalc_matrix_ANN_to_B_F(NS, NSS, V1, V3, V_e1, V_e3,x,y):
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

def calc_B_F(FRAG, precalc_matrix, NSS):
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
def calc_FRAG(epochs_total, V, V_rel, X1, STR1, STR2, STR3, NO_FRAG, int_bre, NS,NSS, x,y, mask):
    ## FRAG stores the information of NS*NS fragments broken into by NSS*NSS particles. 
    ## The first dimension stores the number of fragments. 
    ## The second dimension is the X1 volume of the fragment. 
    ## The third dimension is the volume of X2
    FRAG = np.zeros((3,NSS,NSS,NS,NS))
    model_1d = load_model(f'model_1d_{epochs_total}.keras')
    model_2dX1 = load_model(f'model_2dX1_{epochs_total}.keras')
    model_2dX2 = load_model(f'model_2dX2_{epochs_total}.keras')
    with open(path_scaler,'rb') as file:
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
    
def test_ANN_to_B_F(NS,S,epochs_total):
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
    
    FRAG = calc_FRAG(epochs_total, V, V_rel, X1, STR1, STR2, STR3, NO_FRAG, int_bre, NS, NSS, x,y, mask)
    precalc_matrix = precalc_matrix_ANN_to_B_F(NS,NSS,V1,V3,V_e1,V_e3,x,y)
    int_B_F, intx_B_F, inty_B_F = calc_B_F(FRAG,precalc_matrix,NSS)
    
    np.savez('int_B_F.npz',
             int_B_F=int_B_F,
             intx_B_F = intx_B_F,
             inty_B_F = inty_B_F)
    
    return int_B_F, intx_B_F, inty_B_F, V, FRAG

def plot_error(results,epochs,num_training):
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
    
    ax1, fig1 = pt.plot_data(epochs_array, results['model_2d']['mse'],fig=fig1,ax=ax1,
                           xlbl='Epochs of Model Training / $-$',
                           ylbl='Results of Validation / $-$',
                           lbl='mse_2d',clr='b',mrk='o')
    ax1, fig1 = pt.plot_data(epochs_array, results['model_1d']['mse'],fig=fig1,ax=ax1,
                            lbl='mse_1d',clr='g',mrk='o')
    
    ax2, fig2 = pt.plot_data(epochs_array, results['model_2d']['mae'],fig=fig2,ax=ax2,
                           xlbl='Epochs of Model Training / $-$',
                           ylbl='Results of Validation / $-$',
                           lbl='mae_2d',clr='b',mrk='o')
    ax2, fig2 = pt.plot_data(epochs_array, results['model_1d']['mae'],fig=fig2,ax=ax2,
                            lbl='mae_1d',clr='g',mrk='o')
    
    ax3, fig3 = pt.plot_data(epochs_array, results['model_2d']['mean_frag_erro'],fig=fig3,ax=ax3,
                           xlbl='Epochs of Model Training / $-$',
                           ylbl='Results of Validation / $-$',
                           lbl='mean_frag_erro_2d',clr='b',mrk='o')
    ax3, fig3 = pt.plot_data(epochs_array, results['model_1d']['mean_frag_erro'],fig=fig3,ax=ax3,
                           lbl='mean_frag_erro_1d',clr='g',mrk='o')

    ax4, fig4 = pt.plot_data(epochs_array, results['model_2d']['mean_x_mass_erro'],fig=fig4,ax=ax4,
                           xlbl='Epochs of Model Training / $-$',
                           ylbl='Results of Validation / $-$',
                           lbl='mean_x_mass_erro_2d',clr='b',mrk='o')
    ax4, fig4 = pt.plot_data(epochs_array, results['model_2d']['mean_y_mass_erro'],fig=fig4,ax=ax4,
                           lbl='mean_y_mass_erro_2d',clr='r',mrk='^')
    ax4, fig4 = pt.plot_data(epochs_array, results['model_1d']['mean_x_mass_erro'],fig=fig4,ax=ax4,
                           lbl='mean_x_mass_erro_1d',clr='g',mrk='o')
    return   

def plot_1d_F(x,NS,test_data,epochs_total,data_index=0,vol_dis=False):
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
    
def plot_2d_F(x,y,NS,test_data,epochs_total,data_index=0,vol_dis=False):
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
    pth = os.path.dirname( __file__ )
    ## 1d model has three input parameters: Volume, NO_FRAG, int_bre
    ## 2d model has seven input parameters: Volume, X1, STR1, STR2, STR,3, NO_FRAG, int_bre
    directory = os.path.join(pth,'../../tests/simulation_data')
    test_directory = os.path.join(pth,'../../tests/test_data')
    path_scaler = os.path.join(pth,'Inputs_scaler.pkl')
    path_all_data = os.path.join(pth,'output_data_volX1.pkl')
    path_all_test = os.path.join(pth,'test_data_volX1.pkl')
    NS = 50
    S = 1.3
    N_GRIDS, N_FRACS = 200, 100
    use_custom_loss = True
    ## value in [0,1)
    weight_loss_FRAG_NUM = 0.001
    
    # all_Inputs, F_norm, data_num = load_all_data(directory)
    # ## Use data other than training data for testing
    # test_Inputs, test_F_norm, test_data_num = load_all_data(test_directory) 
    # B_c,v1,v2,M1_c,M2_c,e1,e2,x,y = generate_grid()
    # all_data, all_prob, all_x_mass, all_y_mass = Outputs_on_grid(path_all_data,path_scaler, all_Inputs, F_norm, data_num,
    #                                                                 B_c,v1,v2,M1_c,M2_c,e1,e2,x,y,test_data=False)
    # test_all_data, _, _, _ = Outputs_on_grid(path_all_test,path_scaler, test_Inputs, test_F_norm, test_data_num,
    #                                                                 B_c,v1,v2,M1_c,M2_c,e1,e2,x,y,test_data=True)
    B_c,v1,v2,M1_c,M2_c,e1,e2,x,y = generate_grid()
    with open(path_all_data, 'rb') as file:
        all_data, all_prob, all_x_mass, all_y_mass = pickle.load(file)
    with open(path_all_test, 'rb') as file:
        test_all_data, _, _, _ = pickle.load(file)
    
    # models = [
    #     ("model_1d", create_model_1d(), all_data[0], test_all_data[0], {'x': x[:-1]}),
    #     ("model_2d", create_model_2dX1X2(), all_data[1], test_all_data[1], {'x': x[:-1], 'y': y[:-1]})
    # ]
    
    epochs = 1
    num_training = 50
    # results = {name: {"mse": [], "mae": [], "mean_frag_erro": [], "mean_x_mass_erro": [], "mean_y_mass_erro": [], "mean_all_mass_erro": []} for name, _, _, _, _ in models}
    # start_time = time.time()
    # for training in range(num_training):
    #     for name, model, train_data, test_data, params in models:
    #         test_res = train_and_evaluate_model(model, name, train_data, test_data, epochs, training, **params)
    #         results[name]["mse"].append(test_res[0])
    #         results[name]["mae"].append(test_res[1])
    #         results[name]["mean_frag_erro"].append(test_res[2])
    #         results[name]["mean_x_mass_erro"].append(test_res[3])
    #         results[name]["mean_y_mass_erro"].append(test_res[4])
    #     print(f"Training {training+1} completed!")
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"The Training of psd-data takes: {elapsed_time} seconds")
    # # write and read results, if needed
    # with open(f'epochs_{epochs*num_training}_weight_{weight_loss_FRAG_NUM}.pkl', 'wb') as f:
    #     pickle.dump(results, f)
        
    with open(f'epochs_{epochs*num_training}_weight_{weight_loss_FRAG_NUM}.pkl', 'rb') as f:
        loaded_res = pickle.load(f)
        
    plot_error(loaded_res, epochs, num_training)    

    # int_B_F, intx_B_F, inty_B_F, V, FRAG = test_ANN_to_B_F(NS,S,epochs_total=10000)
    
    # plot_1d_F(x[:-1], NS, test_all_data[0],data_index=0,vol_dis=False,epochs_total=20)
    # plot_2d_F(x[:-1], y[:-1], NS, test_all_data[1],data_index=0,vol_dis=False,epochs_total=20)