# -*- coding: utf-8 -*-
"""
function of creating and training 1D/2D ANN model
"""
import numpy as np
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Reshape, Lambda, BatchNormalization, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras.saving import register_keras_serializable
import tensorflow as tf

def create_model_1d(NS, use_custom_loss, init_neurons=128, num_layers=2, 
                    dropout_rate=0.5,l1_factor=0.001):
    # The input layer accepts an array of length 3 (combined STR and FRAG)
    # The output layer should now match the flattened shape of the new combined output array
    # model = Sequential([
    #     InputLayer(shape=(3,)),  
    #     Dense(128, activation='relu', kernel_regularizer=regularizers.l1(0.001)),
    #     BatchNormalization(),
    #     Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.001)),
    #     BatchNormalization(),
    #     Dense(NS, activation='softmax'),  # Adjusted for the new output shape
    # ])
    model = Sequential()
    model.add(InputLayer(shape=(3,)))
    neurons = init_neurons
    for i in range(num_layers):
        model.add(Dense(neurons, activation='relu', kernel_regularizer=regularizers.l1(l1_factor)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        neurons = max(1, neurons//2)
    model.add(Dense(NS, activation='softmax'))
    if not use_custom_loss:
        model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    return model

def train_model_1d(model, train_data, test_data, x, mask, print_status, weight_loss_FRAG_NUM=0.01,
                   epochs=100, batch_size=32, optimizer_type='adam', learning_rate=0.0001):
    all_Inputs = train_data[0]
    all_Outputs = train_data[1]
    all_Inputs_scaled = train_data[2]
    if optimizer_type == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer_type == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    # optimizer=Adam()
    train_dataset = tf.data.Dataset.from_tensor_slices((all_Inputs.astype(np.float32), all_Outputs.astype(np.float32), 
                                                        all_Inputs_scaled.astype(np.float32))).batch(batch_size)
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    mask_tf = tf.convert_to_tensor(mask)
    
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    for epoch in range(epochs):
        if print_status:
            print(f'Starting epoch {epoch+1}')
        # Training
        for step, (batch_inputs, batch_outputs, batch_inputs_scaled) in enumerate(train_dataset):
            FRAG_NUM = batch_inputs[:,1]
            loss, grads = compute_loss_and_grads(model, batch_inputs_scaled, batch_outputs, 
                                                       x_tf, None, 
                                                       mask_tf, FRAG_NUM, weight_loss_FRAG_NUM)
            # with tf.GradientTape() as tape:
            #     prediction  = model(batch_inputs_scaled, training=True)
            #     loss = combined_custom_loss(batch_outputs, prediction, FRAG_NUM, x_tf, mask_tf, 
            #                                 alpha=weight_loss_FRAG_NUM, beta=1-weight_loss_FRAG_NUM)
            
            # grads = tape.gradient(loss, model.trainable_weights)
            # grads = [tf.clip_by_value(g, -1.0, 1.0) for g in grads]
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            if print_status:
                print(f'Training step {step+1}, Loss: {loss.numpy().mean()}')
        # evaluate_model
        validate_results = validate_model_1d(model,test_data,x,mask)
        # Early Stopping Check with validate_results[0](mse)
        # if validate_results[0] < best_val_loss:
        #     best_val_loss = validate_results[0]
        #     best_validate_results = validate_results
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         if print_status:
        #             print(f"Early stopping at epoch {epoch+1}")
        #         return best_validate_results
    if print_status:    
        print(f"Training of Echos = {epochs} completed!")
    return validate_results
    
def validate_model_1d(model,test_data,x,mask): 
    test_Inputs = test_data[0]
    test_Outputs = test_data[1]
    test_Inputs_scaled = test_data[2]
    predicted_Outputs = model.predict(test_Inputs_scaled, verbose=0)
    ## Compatible with lower versions of tensorfolw
    all_mse = tf.keras.losses.MeanSquaredError()(test_Outputs, predicted_Outputs).numpy()
    all_mae = tf.keras.losses.MeanAbsoluteError()(test_Outputs, predicted_Outputs).numpy()
    mse = all_mse.mean()
    mae = all_mae.mean()

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

def create_model_2dX1(NS, use_custom_loss, init_neurons=256, num_layers=3, 
                      dropout_rate=0.5,l1_factor=0.001):
    @register_keras_serializable()
    def pad_Outputs_X1(Outputs):
        batch_size = tf.shape(Outputs)[0]
        padding = tf.zeros((batch_size, 1, NS), dtype=Outputs.dtype)
        padde_Outputs = tf.concat([padding, Outputs], axis=1)
        return padde_Outputs
    # The input layer accepts an array of length 7 (combined STR and FRAG)
    # The output layer should now match the flattened shape of the new combined output array
    # model = Sequential([
    #     InputLayer(shape=(7,)),  
    #     Dense(256, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
    #     BatchNormalization(),
    #     Dense(128, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
    #     BatchNormalization(),
    #     Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
    #     BatchNormalization(),
    #     Dense((NS-1)*NS, activation='softmax'),  # Adjusted for the new output shape
    #     Reshape((NS-1, NS)),  # Reshape the output to the desired shape
    #     Lambda(pad_Outputs_X1) # Add zero rows in front of first dimension
    # ])
    model = Sequential()
    model.add(InputLayer(shape=(7,)))
    neurons = init_neurons
    for i in range(num_layers):
        model.add(Dense(neurons, activation='relu', kernel_regularizer=regularizers.l1(l1_factor)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        neurons = max(1, neurons//2)
    model.add(Dense((NS-1)*NS, activation='softmax'))
    model.add(Reshape((NS-1, NS))) 
    model.add(Lambda(pad_Outputs_X1))
    
    if not use_custom_loss:
        model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    return model

def create_model_2dX2(NS, use_custom_loss, init_neurons=256, num_layers=3, 
                      dropout_rate=0.5,l1_factor=0.001):
    @register_keras_serializable()
    def pad_Outputs_X2(Outputs):
        batch_size = tf.shape(Outputs)[0]
        padding = tf.zeros((batch_size, NS, 1), dtype=Outputs.dtype)
        padde_Outputs = tf.concat([padding, Outputs], axis=2)
        return padde_Outputs
    # The input layer accepts an array of length 7 (combined STR and FRAG)
    # The output layer should now match the flattened shape of the new combined output array
    # model = Sequential([
    #     InputLayer(shape=(7,)),  
    #     Dense(256, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
    #     BatchNormalization(),
    #     Dense(128, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
    #     BatchNormalization(),
    #     Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
    #     BatchNormalization(),
    #     Dense(NS*(NS-1), activation='softmax'),  # Adjusted for the new output shape
    #     Reshape((NS, NS-1)),  # Reshape the output to the desired shape
    #     Lambda(pad_Outputs_X2) # Add zero column in front of second dimension
    # ])
    model = Sequential()
    model.add(InputLayer(shape=(7,)))
    neurons = init_neurons
    for i in range(num_layers):
        model.add(Dense(neurons, activation='relu', kernel_regularizer=regularizers.l1(l1_factor)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        neurons = max(1, neurons//2)
    model.add(Dense(NS*(NS-1), activation='softmax'))
    model.add(Reshape((NS, NS-1))) 
    model.add(Lambda(pad_Outputs_X2))
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

def train_model_2d(model, train_data, test_data, x, y, mask, print_status, weight_loss_FRAG_NUM=0.01,
                   epochs=100, batch_size=32, optimizer_type='adam', learning_rate=0.0001):
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
    
    if optimizer_type == 'adam':
        optimizer_X1 = Adam(learning_rate=learning_rate)
        optimizer_X2 = Adam(learning_rate=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer_X1 = SGD(learning_rate=learning_rate)
        optimizer_X2 = SGD(learning_rate=learning_rate)
    elif optimizer_type == 'rmsprop':
        optimizer_X1 = RMSprop(learning_rate=learning_rate)
        optimizer_X2 = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    # optimizer_X1=Adam()
    # optimizer_X2=Adam()
    train_dataset = tf.data.Dataset.from_tensor_slices((
        all_Inputs.astype(np.float32),
        all_Outputs_X1.astype(np.float32),
        all_Outputs_X2.astype(np.float32),
        all_Inputs_scaled.astype(np.float32),
        V.astype(np.float32),
        FRAG_NUM.astype(np.float32)
    )).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    mask_tf = tf.convert_to_tensor(mask)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    for epoch in range(epochs):
        if print_status:
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
            if print_status:
                print(f'Training step {step+1}, Loss_X1: {loss_X1.numpy().mean()}, Loss_X2: {loss_X2.numpy().mean()}')
        # evaluate_model
        validate_results = validate_model_2d(model,test_data,x,y,mask)
        # Early Stopping Check with validate_results[0](mse)
        # if validate_results[0] < best_val_loss:
        #     best_val_loss = validate_results[0]
        #     best_validate_results = validate_results
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         if print_status:
        #             print(f"Early stopping at epoch {epoch+1}")
        #         return best_validate_results
        
    if print_status:
        print(f"Training of Echos = {epochs} completed!")
    return validate_results
    
def validate_model_2d(model,test_data,x,y,mask): 
    test_Inputs = test_data[0]
    test_Outputs_X1 = test_data[1]
    test_Outputs_X2 = test_data[2]
    test_Inputs_scaled = test_data[3]
    model_X1 = model[0]
    model_X2 = model[1]
    predicted_Outputs_X1 = model_X1.predict(test_Inputs_scaled, verbose=0)
    predicted_Outputs_X2 = model_X2.predict(test_Inputs_scaled, verbose=0)
    ## Compatible with lower versions of tensorfolw
    all_mse_X1 = tf.keras.losses.MeanSquaredError()(test_Outputs_X1, predicted_Outputs_X1).numpy()
    all_mae_X1 = tf.keras.losses.MeanAbsoluteError()(test_Outputs_X1, predicted_Outputs_X1).numpy()
    all_mse_X2 = tf.keras.losses.MeanSquaredError()(test_Outputs_X2, predicted_Outputs_X2).numpy()
    all_mae_X2 = tf.keras.losses.MeanAbsoluteError()(test_Outputs_X2, predicted_Outputs_X2).numpy()
    mse = (all_mse_X1+all_mse_X2).mean()
    mae = (all_mae_X1+all_mae_X2).mean()
    
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
    ## Compatible with lower versions of tensorfolw
    loss_kl = tf.keras.losses.KLDivergence()(y_true, y_pred)

    return alpha * loss_custom_all_prob + beta * loss_kl