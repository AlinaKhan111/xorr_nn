# src/train_pipeline.py

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import RMSprop
from src.config import config
from src.preprocessing.data_management import load_dataset, save_model

# Load training data
training_data = load_dataset('train.csv')
X_train = training_data.iloc[:, 0:2].values
Y_train = training_data.iloc[:, 2].values.reshape(-1, 1)

# Hyperparameters
epochs = 700
mb_size = 2

def training_data_generator():
    for i in range(training_data.shape[0] // mb_size):
        X_train_mb = X_train[i * mb_size:(i + 1) * mb_size, :]
        Y_train_mb = Y_train[i * mb_size:(i + 1) * mb_size, :]
        yield X_train_mb, Y_train_mb

def functional_mlp():
    inp = Input(shape=(X_train.shape[1],))
    first_hidden_out = Dense(units=4, activation="relu")(inp)
    second_hidden_out = Dense(units=2, activation="relu")(first_hidden_out)
    nn_out = Dense(units=1, activation="sigmoid")(second_hidden_out)
    return Model(inputs=[inp], outputs=[nn_out])

# Build the model
functional_nn = functional_mlp()

# Define loss function
def binary_cross_entropy_loss(Y_hat, Y_true):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=Y_true, y_pred=Y_hat))

# Optimizer
optimizer = RMSprop()

# Training loop
for e in range(epochs):
    for X_train_mb, Y_train_mb in training_data_generator():
        with tf.GradientTape() as tape:
            Y_pred = functional_nn(X_train_mb, training=True)
            loss_func = binary_cross_entropy_loss(Y_pred, Y_train_mb)
        gradients = tape.gradient(loss_func, functional_nn.trainable_weights)
        optimizer.apply_gradients(zip(gradients, functional_nn.trainable_weights))
    print(f"Epoch #{e + 1}, Loss Function Value = {loss_func}")

# Save the trained model
save_model(functional_nn.get_weights())
