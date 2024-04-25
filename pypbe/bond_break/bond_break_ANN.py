# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:35:00 2024

@author: px2030
"""

import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Flatten, InputLayer
from keras.optimizers import Adam

def load_data(directory):
    data = np.load(directory)
    STR = data['STR']
    FRAG = data['FRAG'].reshape(-1, 1)  # Reshaping for compatibility with neural network input
    int_B_F = data['int_B_F']
    intx_B_B = data['intx_B_B']
    iny_B_F = data['iny_B_F']
    return STR, FRAG, int_B_F, intx_B_B, iny_B_F
    
def create_model(input_shape_str, input_shape_frag, output_shape):
    model = Sequential([
        InputLayer(input_shape=(input_shape_str + input_shape_frag,)),  # Adjust input shape as necessary
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Flatten(),
        Dense(np.prod(output_shape), activation='linear')  # Output layer nodes = product of output dimensions
    ])
    
    model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    return model

def train_model(model, inputs, outputs, epochs=10, batch_size=32):
    # Combine STR and FRAG inputs for training
    combined_inputs = np.hstack((inputs[0], inputs[1]))
    # Reshape outputs if they are not already in the required shape (flattened if needed)
    reshaped_outputs = np.reshape(outputs, (outputs.shape[0], -1))
    history = model.fit(combined_inputs, reshaped_outputs, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return history


if __name__ == '__main__':
    # 使用函数
    directory = 'int_B_F'
    STR, FRAG, int_B_F, intx_B_B, iny_B_F = load_data(directory)
    model = create_model(input_shape_str=3, input_shape_frag=1, output_shape=int_B_F.shape[1:])
    history = train_model(model, (STR, FRAG), np.stack((int_B_F, intx_B_B, iny_B_F), axis=-1))