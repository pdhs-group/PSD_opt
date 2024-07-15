# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:34:57 2024

@author: px2030
"""
import numpy as np
import os,sys
sys.path.insert(0,os.path.join(os.path.dirname( __file__ ),"../.."))
from pypbe.bond_break.bond_break_post import calc_int_BF
from pypbe.utils.func.jit_pop import lam, lam_2d, heaviside
## external package
import math
from scipy.integrate import dblquad
from sklearn.neighbors import KernelDensity
# from sklearn.metrics import mean_squared_error
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, InputLayer, Reshape, Layer
from keras.optimizers import Adam
from keras.saving import register_keras_serializable
# from sklearn.model_selection import train_test_split
import tensorflow as tf
# from tensorflow.keras.losses import MeanSquaredError
# import json
import pickle
## for plotter
import matplotlib.pyplot as plt
import pypbe.utils.plotter.plotter as pt

# Register the custom layer for Keras serialization
@register_keras_serializable()
class ScaledSoftmax(Layer):
    def __init__(self, **kwargs):
        super(ScaledSoftmax, self).__init__(**kwargs)

    def call(self, inputs):
        data, input_features = inputs
        # Extract scale factors from the second column of the input features
        scale_factors = input_features[:, 1] 
        # Apply softmax activation
        softmax_output = tf.nn.softmax(data)
        scaled_output = softmax_output * tf.expand_dims(scale_factors, axis=-1)
        return scaled_output
    
# Function to create a 2D output model
def create_model_2d():
    # The input layer accepts an array of length 7
    input_data = Input(shape=(7,))
    # First dense layer with ReLU activation
    x = Dense(128, activation='relu')(input_data)
    x = Dense(64, activation='relu')(x)
    # Output dense layer to match NS*NS size
    x = Dense(NS * NS)(x)
    # Apply ScaledSoftmax layer
    scaled_softmax_output = ScaledSoftmax()([x, input_data])
    # Reshape the output to (NS, NS)
    output = Reshape((NS, NS))(scaled_softmax_output)
    # Create the Keras Model
    model = Model(inputs=input_data, outputs=output)
    # Compile the model with default loss and metrics
    if not use_custom_loss:
        model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    return model
# Function to create a 1D output model
def create_model_1d():
    # The input layer accepts an array of length 3
    model = Sequential([
        InputLayer(shape=(3,)),  
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(NS, activation='softmax'),  # Adjusted for the new output shape
    ])
    if not use_custom_loss:
        model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    return model

def train_model(model, all_Inputs, all_Outputs, V, X1, mask_V, epochs=100, batch_size=32):
    # Define the optimizer
    optimizer=Adam()
    # Create a TensorFlow dataset from inputs and outputs, batch it
    train_dataset = tf.data.Dataset.from_tensor_slices((all_Inputs.astype(np.float32), all_Outputs.astype(np.float32))).batch(batch_size)
    V_tf = tf.convert_to_tensor(V, dtype=tf.float32)
    X1_tf = tf.convert_to_tensor(X1, dtype=tf.float32)
    mask_V_tf = tf.convert_to_tensor(mask_V)
    for epoch in range(epochs):
        print(f'Starting epoch {epoch+1}')
        # Training loop
        for step, (batch_inputs, batch_outputs) in enumerate(train_dataset):
            if V.ndim == 1:
                FRAG_NUM = batch_inputs[:,1]
                prob_X1 = 1.0 
            else:
                FRAG_NUM = batch_inputs[:,2]
                prob_X1 = batch_inputs[:,1]
            with tf.GradientTape() as tape:
                # Forward pass
                prediction  = model(batch_inputs, training=True)
                # Compute custom loss
                loss = combined_custom_loss(batch_outputs, prediction, FRAG_NUM, prob_X1, V_tf, X1_tf, mask_V_tf, 
                                            alpha=0, beta=1, theta=0)
            # Compute gradients
            grads = tape.gradient(loss, model.trainable_weights)
            # Apply gradients
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            print(f'Training step {step+1}, Loss: {loss.numpy().mean()}')
        
    print(f"Training of Echos = {epochs} completed!")

def custom_loss_all_prob(FRAG_NUM, V, mask_V):
    def loss(y_true, y_pred): 
        if V.ndim == 1:
            pre_all_prob = tf.reduce_sum(tf.where(mask_V, y_pred / (V+1e-20), 0), axis=1)
        else:
            pre_all_prob = tf.reduce_sum(tf.where(mask_V, y_pred / (V+1e-20), 0), axis=[1, 2])
        
        relative_error_all_prob = abs((pre_all_prob - FRAG_NUM)) / FRAG_NUM
        mse_all_prob = tf.reduce_mean(tf.square(relative_error_all_prob))

        return mse_all_prob

    return loss
def custom_loss_mass_xy(prob_X1, X1, mask_V):
    def loss(y_true, y_pred): 
        if X1.ndim == 1:
            pre_all_x_mass = tf.reduce_sum(tf.where(mask_V, y_pred * X1, 0), axis=1)
            relative_error_all_y_mass = 0.0
        else:
            pre_all_x_mass = tf.reduce_sum(tf.where(mask_V, y_pred * X1, 0), axis=[1, 2])
            pre_all_y_mass = tf.reduce_sum(tf.where(mask_V, y_pred * (1-X1), 0), axis=[1, 2])
            relative_error_all_y_mass = abs((1 - prob_X1 - pre_all_y_mass)) / pre_all_y_mass
        
        relative_error_all_x_mass = abs((prob_X1 - pre_all_x_mass)) / pre_all_x_mass
        
        mse_all_x_mass = tf.reduce_mean(tf.square(relative_error_all_x_mass))
        mse_all_y_mass = tf.reduce_mean(tf.square(relative_error_all_y_mass))

        return mse_all_x_mass + mse_all_y_mass

    return loss

def combined_custom_loss(y_true, y_pred, FRAG_NUM, prob_X1, V, X1, mask_V, alpha=0.5, beta=0.5, theta=0.5):
    custom_loss_func_all_prob = custom_loss_all_prob(FRAG_NUM, V, mask_V)
    loss_custom_all_prob = custom_loss_func_all_prob(y_true, y_pred)
    custom_loss_func_mass_xy = custom_loss_mass_xy(prob_X1, X1, mask_V)
    loss_custom_mass_xy = custom_loss_func_mass_xy(y_true, y_pred)
    loss_mse = tf.keras.losses.mean_squared_error(y_true, y_pred)

    return alpha * loss_custom_all_prob + beta * loss_mse + theta * loss_custom_mass_xy

def train_and_evaluate_model(model, model_name, train_data, test_data, epochs, x, y=None):
    if y is not None:
        V = np.zeros((len(x),len(y)))
        X1 = np.zeros_like(V)
        for i, v_x in enumerate(x):
            for j, v_y in enumerate(y):
                if i + j != 0:
                    V[i,j] = v_x + v_y
                    X1[i,j] = v_x / V[i,j]
        mask = np.ones(V.shape, dtype=bool)
        mask[0, 0] = False
    else:
        V = x
        X1 = np.ones_like(V)
        mask = np.ones(V.shape, dtype=bool)
        mask[0] = False
    # train model
    if use_custom_loss:
        train_model(model, train_data[0], train_data[1], epochs=epochs, V=V, X1=X1, mask_V=mask)
    else:
        model.fit(train_data[0], train_data[1], epochs=epochs, batch_size=32, validation_split=0.2)   
  
    model.save(model_name + '.keras')
    # evaluate_model
    test_res = predictions_test(model_name + '.keras', test_data[0], test_data[1], V=V, X1=X1, mask=mask)
    return test_res

def predictions_test(model_name,test_all_Inputs, test_all_Outputs, V, X1, mask): 
    model = load_model(model_name)
    predicted_all_Outputs = model.predict(test_all_Inputs)
    if use_custom_loss:
        all_mse = tf.keras.losses.mean_squared_error(test_all_Outputs, predicted_all_Outputs).numpy()
        all_mae = tf.keras.losses.mean_absolute_error(test_all_Outputs, predicted_all_Outputs).numpy()
        mse = all_mse.mean()
        mae = all_mae.mean()
    else:
        mse, mae = model.evaluate(test_all_Inputs, test_all_Outputs)
    length = test_all_Inputs.shape[0]
    pre_all_prob = np.zeros(length)
    pre_all_x_mass = np.zeros(length)
    pre_all_y_mass = np.zeros(length)
    ## In the case of 1d, test_all_Inputs[i,1] represents the number of fragments
    ## In the case of 2d, test_all_Inputs[i,1] represents the Volume fraction of X1,
    ## test_all_Inputs[i,2] represents the number of fragments
    if V.ndim == 1:
        true_all_prob =  test_all_Inputs[:,1]
        true_all_x_mass = np.ones_like(true_all_prob)
    else:
        true_all_prob =  test_all_Inputs[:,2]
        true_all_x_mass = test_all_Inputs[:,1]
    true_all_y_mass = (1 - true_all_x_mass)
    for i in range(length):
        pre_all_prob[i] = np.sum(predicted_all_Outputs[i,mask]/ V[mask])
        pre_all_x_mass[i] = np.sum(predicted_all_Outputs[i,mask] * X1[mask])
        pre_all_y_mass[i] = np.sum(predicted_all_Outputs[i,mask] * (1-X1[mask]))
            
    mean_all_prob = (abs(true_all_prob - pre_all_prob) / true_all_prob).mean()
    mean_all_x_mass = (abs(true_all_x_mass - pre_all_x_mass) / true_all_x_mass).mean()
    if V.ndim == 1:
        mean_all_y_mass = 0.0
        mean_all_mass = mean_all_x_mass
    else:
        mean_all_y_mass = (abs(true_all_y_mass - pre_all_y_mass) / true_all_y_mass).mean()
        mean_all_mass = np.sum(abs(1.0-(pre_all_x_mass+pre_all_y_mass)))

    return mse, mae, mean_all_prob, mean_all_x_mass, mean_all_y_mass, mean_all_mass

# %% POST PROCESSING  
def plot_error(results,epochs,num_training):
    epochs_array = np.arange(epochs, epochs*num_training+1, epochs)
    pt.close()
    pt.plot_init(scl_a4=1,figsze=[12.8,6.4*1.5],lnewdth=0.8,mrksze=5,use_locale=True,scl=1.2)
    ax1, fig1 = pt.plot_data(epochs_array, results['model_2d']['mse'],
                           xlbl='Epochs of Model Training / $-$',
                           ylbl='Results of Validation / $-$',
                           lbl='mse',clr='b',mrk='o')
    ax2, fig2 = pt.plot_data(epochs_array, results['model_2d']['mae'],
                           xlbl='Epochs of Model Training / $-$',
                           ylbl='Results of Validation / $-$',
                           lbl='mae',clr='b',mrk='o')
    ax3, fig3 = pt.plot_data(epochs_array, results['model_2d']['mean_frag_erro'],
                           xlbl='Epochs of Model Training / $-$',
                           ylbl='Results of Validation / $-$',
                           lbl='mean_frag_erro',clr='b',mrk='o')
    fig4=plt.figure()
    ax4=fig4.add_subplot(1,1,1)
    ax4, fig4 = pt.plot_data(epochs_array, results['model_2d']['mean_x_mass_erro'],fig=fig4,ax=ax4,
                           xlbl='Epochs of Model Training / $-$',
                           ylbl='Results of Validation / $-$',
                           lbl='mean_x_mass_erro',clr='b',mrk='o')
    ax4, fig4 = pt.plot_data(epochs_array, results['model_2d']['mean_y_mass_erro'],fig=fig4,ax=ax4,
                           xlbl='Epochs of Model Training / $-$',
                           ylbl='Results of Validation / $-$',
                           lbl='mean_y_mass_erro',clr='r',mrk='^')
    ax5, fig5 = pt.plot_data(epochs_array, results['model_2d']['mean_all_mass_erro'],
                           xlbl='Epochs of Model Training / $-$',
                           ylbl='Results of Validation / $-$',
                           lbl='mean_all_mass_erro',clr='b',mrk='o')
    return    
# %% MAIN
if __name__ == '__main__':
    ## 1d model has three input parameters: Volume, NO_FRAG, int_bre
    ## 2d model has seven input parameters: Volume, X1, STR1, STR2, STR,3, NO_FRAG, int_bre
    psd = 'direct'
    int_method = 'MC'
    directory = '../../tests/simulation_data'
    test_directory = '../../tests/test_data'
    path_all_data = 'output_data_volX1.pkl'
    path_all_test = 'test_data_volX1.pkl'
    NS = 30
    S = 2
    N_GRIDS, N_FRACS = 200, 100
    max_NO_FRAG = 8
    max_len = max_NO_FRAG * N_GRIDS * N_FRACS
    use_custom_loss = True
    
    with open(path_all_data, 'rb') as file:
        all_data, all_prob, all_x_mass = pickle.load(file)
        
    with open(path_all_test, 'rb') as file:
        test_all_data, _, _ = pickle.load(file)
    
    models = [
        ("model_1d", create_model_1d(), all_data[0], test_all_data[0], {'x': x[:-1]}),
        ("model_2d", create_model_2d(), all_data[1], test_all_data[1], {'x': x[:-1], 'y': y[:-1]})
    ]
    epochs = 2
    num_training = 1
    results = {name: {"mse": [], "mae": [], "mean_frag_erro": [], "mean_x_mass_erro": [], "mean_y_mass_erro": [], "mean_all_mass_erro": []} for name, _, _, _, _ in models}
    
    for training in range(num_training):
        for name, model, train_data, test_data, params in models:
            test_res = train_and_evaluate_model(model, name, train_data, test_data, epochs, **params)
            results[name]["mse"].append(test_res[0])
            results[name]["mae"].append(test_res[1])
            results[name]["mean_frag_erro"].append(test_res[2])
            results[name]["mean_x_mass_erro"].append(test_res[3])
            results[name]["mean_y_mass_erro"].append(test_res[4])
            results[name]["mean_all_mass_erro"].append(test_res[5])

    ## write and read results, if needed
    with open(f'epochs_{epochs}_num_{num_training}_vol.pkl', 'wb') as f:
        pickle.dump(results, f)
        
    with open(f'epochs_{epochs}_num_{num_training}_vol.pkl', 'rb') as f:
        loaded_res = pickle.load(f)
        
    plot_error(results, epochs, num_training)    

    # int_B_F, intx_B_F, inty_B_F, V = test_ANN_to_B_F(NS,S)