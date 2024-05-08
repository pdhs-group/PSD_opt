# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:34:57 2024

@author: px2030
"""
import numpy as np
import os
from sklearn.neighbors import KernelDensity
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Reshape
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError

def load_all_data(max_len,directory,kde_gitter):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]
    all_Inputs = np.zeros((len(files), 6))
    all_kde = np.zeros((len(files),NS))
    
    for i, file in enumerate(files):
        filename = os.path.basename(file)
        parsed_variables = np.array(filename[:-4].split('_'),dtype=float)
        F_tem = np.load(file,allow_pickle=True)
        shape = F_tem.shape[0]
        F_norm = np.zeros((shape,3))
        A = parsed_variables[0]
        ## Normalize fragments volume
        F_norm[:,0] = F_tem[:,0] * F_tem[:,1] / A
        F_norm[:,1] = F_tem[:,0] * F_tem[:,2] / A
        F_norm[:,2] = F_tem[:,3]
        kde = KernelDensity(kernel='gaussian', bandwidth='scott').fit(F_norm[:,:2])
        ## Each test produces NO_FRAG fragments
        kde_dense = np.exp(kde.score_samples(kde_gitter)) / parsed_variables[2]
        
        all_Inputs[i,:] = parsed_variables
        all_kde[i,:] = kde_dense
        
    return all_Inputs, all_kde
    
def create_model(max_len):
    # The input layer accepts an array of length 4 (combined STR and FRAG)
    # The output layer should now match the flattened shape of the new combined output array
    model = Sequential([
        InputLayer(input_shape=(6,)),  
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(NS, activation='linear'),  # Adjusted for the new output shape
        # Reshape((max_len, 4))  # Reshape the output to the desired shape
    ])
    
    # model.compile(optimizer=Adam(), loss=combined_custom_loss(V_value, NO_FRAG_value, alpha=0.01, beta=0.99), metrics=['mae'])
    model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    return model

def train_model(model, all_Inputs, all_Outputs, epochs=100, batch_size=32):
    history = model.fit(all_Inputs, all_Outputs, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return history

def custom_loss(V, NO_FRAG):
    def loss(y_true, y_pred):
        
        # print("y_pred shape:", tf.shape(y_pred))
        BF = y_pred[..., 0]
        BFX = y_pred[..., 1]
        BFY = y_pred[..., 2]
        
        BF_sum = tf.reduce_sum(BF, axis=[1, 2])
        loss_bf = tf.reduce_mean(tf.square(BF_sum - NO_FRAG))  
       
        BFX_sum = tf.reduce_sum(BFX, axis=[1, 2])
        BFY_sum = tf.reduce_sum(BFY, axis=[1, 2])
        BFXY_sum = BFX_sum + BFY_sum  
        loss_bfxy = tf.reduce_mean(tf.square(BFXY_sum - V))
        
        return loss_bf + loss_bfxy
    return loss

def combined_custom_loss(V, NO_FRAG, alpha=0.5, beta=0.5):
    mse_loss = MeanSquaredError()
    custom_loss_func = custom_loss(V, NO_FRAG)
    
    def loss(y_true, y_pred):
        loss_custom = custom_loss_func(y_true, y_pred)
        loss_mse = mse_loss(y_true, y_pred)
        
        # 组合损失，alpha 和 beta 是权重参数，可以根据需要调整
        return alpha * loss_custom + beta * loss_mse

    return loss

def predictions_test(model,test_inputs,eva_inputs=None,eva_outputs=None):
    predicted_outputs = model.predict(test_inputs)
    int_B_F = predicted_outputs[0,:,:,:,:,0]
    intx_B_F = predicted_outputs[0,:,:,:,:,1]
    inty_B_F = predicted_outputs[0,:,:,:,:,2]
    pre_FRAG = int_B_F.sum(axis=0).sum(axis=0)
    pre_V = intx_B_F.sum(axis=0).sum(axis=0) + inty_B_F.sum(axis=0).sum(axis=0)
    
    if eva_inputs is not None and eva_outputs is not None:
        test_loss, test_mae = model.evaluate(eva_inputs, eva_outputs)
        return pre_FRAG, pre_V, test_loss, test_mae
    return pre_FRAG, pre_V
    
if __name__ == '__main__':
    # 使用函数
    directory = '../../tests/simulation_data'
    NS = 30
    S = 2
    N_GRIDS, N_FRACS = 200, 100
    max_NO_FRAG = 8
    max_len = max_NO_FRAG * N_GRIDS * N_FRACS
    
    kde_gitter = np.zeros((NS,2))
    for i in range(NS):
            kde_gitter[i,0] = S ** (i)
    kde_gitter[:,0] /= kde_gitter[-1,0]
    kde_gitter[:,1] = 1 - kde_gitter[:,0]
    
    all_Inputs, all_Outputs = load_all_data(max_len,directory,kde_gitter)
    model = create_model(max_len)
    model.summary()
    history = train_model(model, all_Inputs, all_Outputs)
    
    # test_inputs = np.array([0.5,0.5,1.0,4]).reshape(1, 4)
    # pre_FRAG, pre_V=predictions_test(model,test_inputs)