# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:35:00 2024

@author: px2030
"""

import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Reshape
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError

def load_data(file_path):
    data = np.load(file_path)
    STR = data['STR']
    NO_FRAG = data['NO_FRAG']  # Reshaping for compatibility with neural network input
    Outputs = np.stack((data['int_B_F'], data['intx_B_F'], data['inty_B_F']), axis=-1)
    return STR, NO_FRAG, Outputs

def load_all_data(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('int_B_F.npz')]
    all_STR = []
    all_NO_FRAG = []
    all_Outputs = []
    for file in files:
        STR, NO_FRAG, Outputs = load_data(file)
        all_STR.append(STR)
        all_NO_FRAG.append(NO_FRAG)
        all_Outputs.append(Outputs)
    # Convert lists to numpy arrays and stack them appropriately
    return np.vstack(all_STR), np.hstack(all_NO_FRAG), np.stack(all_Outputs, axis=0)
    
def create_model(V_value,NO_FRAG_value):
    # The input layer accepts an array of length 4 (combined STR and FRAG)
    # The output layer should now match the flattened shape of the new combined output array
    model = Sequential([
        InputLayer(input_shape=(4,)),  # Combined STR and FRAG input
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(NS * NS * NS * NS * 3, activation='linear'),  # Adjusted for the new output shape
        Reshape((NS, NS, NS, NS, 3))  # Reshape the output to the desired shape
    ])
    
    model.compile(optimizer=Adam(), loss=combined_custom_loss(V_value, NO_FRAG_value, alpha=0.01, beta=0.99), metrics=['mae'])
    # model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    return model

def train_model(model, STR, NO_FRAG, Outputs, epochs=100, batch_size=32):
    # Convert FRAG from integer to array and combine with STR
    combined_inputs =np.column_stack((STR, NO_FRAG))
    history = model.fit(combined_inputs, Outputs, epochs=epochs, batch_size=batch_size, validation_split=0.2)
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
    directory = 'train_data'
    NS = 7
    STR, NO_FRAG, Outputs = load_all_data(directory)
    V = Outputs[0,:,:,:,:,1].sum(axis=0).sum(axis=0) + Outputs[0,:,:,:,:,2].sum(axis=0).sum(axis=0)
    V = tf.constant(V, dtype=tf.float32)
    NO_FRAG_value = 4.0
    NO_FRAG_value = tf.broadcast_to(NO_FRAG_value, tf.shape(V))
    model = create_model(V,NO_FRAG_value)
    model.summary()
    history = train_model(model, STR, NO_FRAG, Outputs)
    
    test_inputs = np.array([0.5,0.5,1.0,4]).reshape(1, 4)
    pre_FRAG, pre_V=predictions_test(model,test_inputs)