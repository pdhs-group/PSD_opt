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
from scipy.integrate import dblquad
from sklearn.neighbors import KernelDensity
# from sklearn.metrics import mean_squared_error
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, InputLayer, Reshape, Layer
from keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
import tensorflow as tf
# from tensorflow.keras.losses import MeanSquaredError
# import json
import pickle

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
    ## outward by one unit with value 0
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

def Outputs_on_grid(path_all_data, all_Inputs, F_norm, data_num,B_c,v1,v2,M1_c,M2_c,e1,e2,x,y):
    Inputs_2d = []
    Inputs_1d = []
    Outputs_2d = []
    Outputs_1d = []
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
        else:
            tem_Outputs, all_prob[i], all_x_mass[i], all_y_mass[i]  = direct_psd_2d(F_norm_tem,NO_FRAG,B_c,v1,v2,M1_c,M2_c,e1,e2,x,y)   
            Outputs_2d.append(tem_Outputs)
            Inputs_2d.append(all_Inputs[i,:])
    all_data = [[np.array(Inputs_1d),np.array(Outputs_1d)],
                [np.array(Inputs_2d),np.array(Outputs_2d)]]
    
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
    B /= NO_FRAG 
    prob = B.sum()
    x_mass = np.sum(B * x[:-1])
    return B, prob, x_mass
        
def direct_psd_2d(F_norm, NO_FRAG,B_c,v1,v2,M1_c,M2_c,e1,e2,x,y):
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
    B /= NO_FRAG 
    prob = B.sum()
    x_mass = np.sum(B * np.outer(x[:-1], np.ones(30)))
    y_mass = np.sum(B * np.outer(np.ones(30), y[:-1]))
    return B, prob, x_mass, y_mass
# %% DATA PROCESSING KDE
def generate_grid_kde():
    kde_value = np.zeros(NS+1)
    grid = []
    # Define grid with geo-grid
    for i in range(NS):
            kde_value[i+1] = S ** (i)
    kde_value[:] /= kde_value[-1]
    
    ## Use the sin function to define a non-uniform grid. 
    ## This grid is denser on the boundary.
    # uniform_data = np.arange(0, NS+1, dtype=float)/NS
    # kde_value = np.sin(np.pi * uniform_data - np.pi/2) / 2 + 0.5
    for x in kde_value:
        for y in kde_value:
            if 0 < x+y <= 1 :
                grid.append((x,y))
    grid = np.array(grid)
    return grid

def Outputs_on_kde_grid(all_Inputs, F_norm, data_num, grid):
    all_Outputs = []
    all_prob = np.zeros(data_num)
    all_x_mass = np.zeros(data_num)
    all_y_mass = np.zeros(data_num)
    for i in range(data_num):
        X1 = all_Inputs[i,1]
        NO_FRAG = all_Inputs[i,2]
        F_norm_tem = F_norm[i]
        Outputs, all_prob[i], all_x_mass[i], all_y_mass[i] = kde_psd(F_norm_tem, X1, grid,NO_FRAG)
        all_Outputs.append(Outputs)
    return np.array(all_Outputs), all_prob, all_x_mass, all_y_mass

def kde_psd(F_norm, X1, grid, NO_FRAG):
    kde = KernelDensity(kernel='epanechnikov', bandwidth='scott',atol=0, rtol=0).fit(F_norm[:,:2])
    kde_dense = np.exp(kde.score_samples(grid)) 
    prob, x_mass, y_mass = prob_integral(kde,X1, NO_FRAG,method=int_method)
    return kde_dense, prob, x_mass, y_mass
 
def prob_integral(kde,X1,NO_FRAG,method='dblquad'):
    if method == 'dblquad':
        args = (kde,NO_FRAG)
        prob, err = dblquad (prob_func,0,X1,0,1-X1,args=args)
        x_mass = dblquad (x_mass_func,0,X1,0,1-X1,args=args)[0] / prob * NO_FRAG
        y_mass = dblquad (y_mass_func,0,X1,0,1-X1,args=args)[0] / prob * NO_FRAG
        
    elif method == 'MC':
        if X1 != 0:
            samples = np.random.uniform(0, 1, (10000, 2))
            samples = samples[(samples[:, 0] < X1) & (samples[:, 1] < (1 - X1))] 
            prob_densities = [prob_func(x, y, kde, NO_FRAG) for x, y in samples]
            x_mass_densities = [x_mass_func(x, y, kde, NO_FRAG) for x, y in samples]
            y_mass_densities = [y_mass_func(x, y, kde, NO_FRAG) for x, y in samples]
            prob = np.mean(prob_densities) * X1 * (1-X1)
            x_mass = np.mean(x_mass_densities) * X1 * (1-X1) / prob * NO_FRAG
            y_mass = np.mean(y_mass_densities) * X1 * (1-X1) / prob * NO_FRAG
        else:
            samples = np.random.uniform(0, 1, 10000)
            prob_densities = [prob_func(0, y, kde, NO_FRAG) for y in samples]
            y_mass_densities = [y_mass_func(0, y, kde, NO_FRAG) for y in samples]
            prob = np.mean(prob_densities)
            x_mass = 0
            y_mass = np.mean(y_mass_densities) / prob * NO_FRAG
    return prob, x_mass, y_mass
    
def prob_func(x,y,kde,NO_FRAG):
    kde_input = np.array((x,y)).reshape(1,2)
    return np.exp(kde.score_samples(kde_input))

def x_mass_func(x,y,kde,NO_FRAG):
    kde_input = np.array((x,y)).reshape(1,2)
    return np.exp(kde.score_samples(kde_input))[0] * x

def y_mass_func(x,y,kde,NO_FRAG):
    kde_input = np.array((x,y)).reshape(1,2)
    return np.exp(kde.score_samples(kde_input))[0] * y
# %% MODEL TRAINING  
class CustomLayer(Layer):
    def __init__(self, x, y, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.x = x
        self.y = y

    def call(self, inputs):
        return inputs, self.x, self.y
   
def create_model_2d():
    # The input layer accepts an array of length 7 (combined STR and FRAG)
    # The output layer should now match the flattened shape of the new combined output array
    model = Sequential([
        InputLayer(shape=(7,)),  
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(NS*NS, activation='softmax'),  # Adjusted for the new output shape
        Reshape((NS, NS)),  # Reshape the output to the desired shape
        # custom_layer
    ])
    
    return model

def create_model_1d():
    # The input layer accepts an array of length 4 (combined STR and FRAG)
    # The output layer should now match the flattened shape of the new combined output array
    model = Sequential([
        InputLayer(shape=(3,)),  
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(NS, activation='softmax'),  # Adjusted for the new output shape
        # Reshape((NS, NS))  # Reshape the output to the desired shape
        # custom_layer
    ])

    return model

def train_model(model, all_Inputs, all_Outputs, x, y, epochs=100, batch_size=32, optimizer=Adam()):
    # 
    train_dataset = tf.data.Dataset.from_tensor_slices((all_Inputs.astype(np.float32), all_Outputs.astype(np.float32))).batch(batch_size)
    optimizer = Adam()
    # Convert numpy array to tensorflow format for calculation of loss function
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    if y is not None:
        y_tf = tf.convert_to_tensor(y, dtype=tf.float32)
    else:
        y_tf = None
    for epoch in range(epochs):
        print(f'Starting epoch {epoch+1}')
        # Training
        for step, (batch_inputs, batch_outputs) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                prediction  = model(batch_inputs, training=True)
                loss = combined_custom_loss(batch_outputs, prediction, batch_inputs, x_tf, y_tf, alpha=0.5, beta=0.5)
            
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            print(f'Training step {step+1}, Loss: {loss.numpy().mean()}')
        
    print(f"Training of Echos = {epochs} completed!")
    # history = model.fit(all_Inputs, all_Outputs, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    # return history

def custom_loss_2d(inputs, x, y):
    def loss(y_true, y_pred):
        outer_x = tf.tensordot(x, tf.ones_like(y), axes=0)
        outer_y = tf.tensordot(tf.ones_like(x), y, axes=0)
        
        pre_all_x_mass = tf.reduce_sum(y_pred * outer_x, axis=[1, 2])
        pre_all_y_mass = tf.reduce_sum(y_pred * outer_y, axis=[1, 2])

        true_all_x_mass = 1.0 * inputs[:, 1] / inputs[:, 2]
        true_all_y_mass = 1.0 * (1 - inputs[:, 1]) / inputs[:, 2]

        relative_error_x_mass = (pre_all_x_mass - true_all_x_mass) / true_all_x_mass
        relative_error_y_mass = (pre_all_y_mass - true_all_y_mass) / true_all_y_mass

        mse_all_x_mass = tf.reduce_mean(tf.square(relative_error_x_mass))
        mse_all_y_mass = tf.reduce_mean(tf.square(relative_error_y_mass))

        return mse_all_x_mass + mse_all_y_mass
    return loss

def custom_loss_1d(inputs, x):
    def loss(y_true, y_pred):
        pre_all_mass = tf.reduce_sum(y_pred * x, axis=1)

        true_all_mass = 1.0 / inputs[:, 1]

        relative_error_mass = (pre_all_mass - true_all_mass) / true_all_mass

        mse_all_mass = tf.reduce_mean(tf.square(relative_error_mass))

        return mse_all_mass
    return loss

def combined_custom_loss(y_true, y_pred, inputs, x, y, alpha=0.5, beta=0.5):
    if y is None:
        custom_loss_func = custom_loss_1d(inputs, x)
    else:
        custom_loss_func = custom_loss_2d(inputs, x, y)
    
    loss_custom = custom_loss_func(y_true, y_pred)
    loss_mse = tf.keras.losses.mean_squared_error(y_true, y_pred)

    return alpha * loss_custom + beta * loss_mse

def train_and_evaluate_model(model, model_name, train_data, test_data, epochs, x=None, y=None):
    # train model
    train_model(model, train_data[0], train_data[1], epochs=epochs, x=x, y=y)
    model.save(model_name + '.keras')
    # evaluate_model
    test_res = predictions_test(model_name + '.keras', test_data[0], test_data[1], x=x, y=y)
    return test_res

def predictions_test(model_name,test_all_Inputs, test_all_Outputs, x=None, y=None): 
    model = load_model(model_name)
    predicted_all_Outputs = model.predict(test_all_Inputs)
    all_mse = tf.keras.losses.mean_squared_error(test_all_Outputs, predicted_all_Outputs).numpy()
    all_mae = tf.keras.losses.mean_absolute_error(test_all_Outputs, predicted_all_Outputs).numpy()
    mse = all_mse.mean()
    mae = all_mae.mean()
    # test_mse, test_mae = model.evaluate(test_all_Inputs, test_all_Outputs)
    length = test_all_Inputs.shape[0]
    pre_all_prob = np.zeros(length)
    pre_all_x_mass = np.zeros(length)
    pre_all_y_mass = np.zeros(length)
    true_all_x_mass = np.zeros(length)
    true_all_y_mass = np.zeros(length)
    for i in range(length):
        if y is None:
            pre_all_prob[i] = np.sum(predicted_all_Outputs[i,:])
            pre_all_x_mass[i] = np.sum(predicted_all_Outputs[i,:] * x)
            ## In the case of 1d, test_all_Inputs[i,1] represents the number of fragments
            true_all_x_mass[i] = 1.0 / test_all_Inputs[i,1]
        else:
            pre_all_prob[i] = np.sum(predicted_all_Outputs[i,:,:])
            pre_all_x_mass[i] = np.sum(predicted_all_Outputs[i,:,:] * np.outer(x, np.ones(30)))
            pre_all_y_mass[i] = np.sum(predicted_all_Outputs[i,:,:] * np.outer(np.ones(30), y))
            ## In the case of 1d, test_all_Inputs[i,1] represents the Volume fraction of X1,
            ## test_all_Inputs[i,2] represents the number of fragments
            true_all_x_mass[i] = 1.0 * test_all_Inputs[i,1]  / test_all_Inputs[i,2]
            true_all_y_mass[i] = 1.0 * (1 - test_all_Inputs[i,1])  / test_all_Inputs[i,2]
    mean_all_x_mass = (abs(true_all_x_mass - pre_all_x_mass) / true_all_x_mass).mean()
    if y is not None:
        mean_all_y_mass = (abs(true_all_y_mass - pre_all_y_mass) / true_all_y_mass).mean()
    else:
        mean_all_y_mass = 0.0
    return mse, mae, pre_all_prob, pre_all_x_mass, pre_all_y_mass, mean_all_x_mass, mean_all_y_mass

# %% POST PROCESSING  

# %% MAIN   
if __name__ == '__main__':
    psd = 'direct'
    int_method = 'MC'
    directory = '../../tests/simulation_data'
    test_directory = '../../tests/test_data'
    path_all_data = 'output_data.pkl'
    path_all_test = 'test_data.pkl'
    NS = 30
    S = 2
    N_GRIDS, N_FRACS = 200, 100
    max_NO_FRAG = 8
    max_len = max_NO_FRAG * N_GRIDS * N_FRACS
    
    # all_Inputs, F_norm, data_num = load_all_data(directory)
    # ## Use data other than training data for testing
    # test_all_Inputs, test_F_norm, test_data_num = load_all_data(test_directory) 
    # if psd == 'direct':
    #     B_c,v1,v2,M1_c,M2_c,e1,e2,x,y = generate_grid()
    #     all_data, all_prob, all_x_mass, all_y_mass = Outputs_on_grid(path_all_data, all_Inputs, F_norm, data_num,
    #                                                                     B_c,v1,v2,M1_c,M2_c,e1,e2,x,y)
    #     test_all_data, _, _, _ = Outputs_on_grid(path_all_test, test_all_Inputs, test_F_norm, test_data_num,
    #                                                                     B_c,v1,v2,M1_c,M2_c,e1,e2,x,y)
    # elif psd == 'KDE':
    #     grid = generate_grid_kde()
    #     all_Outputs, all_prob, all_x_mass, all_y_mass = Outputs_on_kde_grid(all_Inputs, F_norm, data_num, grid)
    
    
    B_c,v1,v2,M1_c,M2_c,e1,e2,x,y = generate_grid()
    with open(path_all_data, 'rb') as file:
        all_data, all_prob, all_x_mass, all_y_mass = pickle.load(file)
        
    with open(path_all_test, 'rb') as file:
        test_all_data, _, _, _ = pickle.load(file)
    all_mass = all_x_mass + all_y_mass

    
    models = [
        ("model_1d", create_model_1d(), all_data[0], test_all_data[0], {'x': x[:-1]}),
        ("model_2d", create_model_2d(), all_data[1], test_all_data[1], {'x': x[:-1], 'y': y[:-1]})
    ]
    
    epochs = 2
    num_training = 25
    results = {name: {"mse": [], "mae": [], "mean_x_mass": [], "mean_y_mass": []} for name, _, _, _, _ in models}
    
    for training in range(num_training):
        for name, model, train_data, test_data, params in models:
            test_res = train_and_evaluate_model(model, name, train_data, test_data, epochs, **params)
            results[name]["mse"].append(test_res[0])
            results[name]["mae"].append(test_res[1])
            results[name]["mean_x_mass"].append(test_res[5])
            results[name]["mean_y_mass"].append(test_res[6])
    ## write and read results, if needed
    with open(f'epochs_{epochs}_num_{num_training}.pkl', 'wb') as f:
        pickle.dump(results, f)
        
    with open(f'epochs_{epochs}_num_{num_training}.pkl', 'rb') as f:
        loaded_res = pickle.load(f)