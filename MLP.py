import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from MyFunctions import *
import json
import pickle

def MLP(X_train, X_test, T_train, T_test, data):
    """Back propagation based on the architecture HNF has constructed"""
    train_accuracy_lists = []
    train_NME_lists = []
    test_accuracy_lists = []
    test_NME_lists = []
    m = X_train.shape[1]
    P = X_train.shape[0]
    Q= T_train.shape[0]
    data = data
    learning_rate = 10**(-6)
    Epoch_num = 10
    batchSize = 32
    Layer_Num = 1

    iteration_num = round(m/batchSize)

    # _logger.info("Read parameters by HNF")
    # parameters_path = "./parameters/"

    ######################################################################################
    ####################        Tensorflow v2       ######################################
    ######################################################################################
    inputs = keras.Input(shape=X_train.shape[0])
    h = layers.Dense(500, activation="relu", name="output_layer")(inputs)
    outputs = layers.Dense(Q)(h)

    model = Model(inputs=inputs, outputs=outputs)
    # model.summary()

    model.compile(loss="MSE", optimizer="adam", metrics=["MeanSquaredError"])
    model.fit(X_train.T, T_train.T, batch_size=batchSize, epochs=Epoch_num, validation_split=0.1, verbose=0)

    # score = model.evaluate(X_test.T, T_test.T, verbose=0)
    # print("Test loss:", score[0])
    # print("Test accuracy:", score[1])

    t_hat = model.predict(X_train.T)
    t_hat_test = model.predict(X_test.T)

    return compute_nme(t_hat.T,T_train), compute_nme(t_hat_test.T,T_test), compute_mse(t_hat.T,T_train), compute_mse(t_hat_test.T,T_test)


        
