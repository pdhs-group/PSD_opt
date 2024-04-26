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
    return np.vstack(all_STR), np.hstack(all_NO_FRAG), np.concatenate(all_Outputs, axis=0)
    
def create_model():
    # The input layer accepts an array of length 4 (combined STR and FRAG)
    # The output layer should now match the flattened shape of the new combined output array
    model = Sequential([
        InputLayer(input_shape=(4,)),  # Combined STR and FRAG input
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Flatten(),
        Dense(NS * NS * NS * NS * 3, activation='linear')  # Adjusted for the new output shape
    ])
    
    model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    return model

def train_model(model, STR, NO_FRAG, Outputs, epochs=10, batch_size=32):
    # Convert FRAG from integer to array and combine with STR
    combined_inputs = np.array([np.append(STR, NO_FRAG)])
    # Reshape outputs to flatten them, assuming outputs are already combined in the correct shape
    reshaped_outputs = Outputs.reshape(-1, NS * NS * NS * NS * 3)
    history = model.fit(combined_inputs, reshaped_outputs, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return history

if __name__ == '__main__':
    # 使用函数
    directory = 'train_data'
    NS = 7
    STR, NO_FRAG, Outputs = load_all_data(directory)
    model = create_model()
    history = train_model(model, STR, NO_FRAG, Outputs)