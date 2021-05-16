from re import VERBOSE
from MyFunctions import *
import tensorflow as tf
from tensorflow.keras import layers


class CNNModel:
    def __init__(self, num_classes=10, input_shape=(28, 28, 1)):
        self.num_classes = num_classes
        self.input_shape = input_shape

    def create_cnn_model(self):
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=self.input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )
        return model

def my_CNN( X_tr, X_ts, T_train, T_test):
    num_classes = 10
    input_shape = (28, 28, 1)

    X_tr = tf.reshape(X_tr.T, (X_tr.shape[1],28,28,1))
    X_ts = tf.reshape(X_ts.T, (X_ts.shape[1],28,28,1))

    model = CNNModel(num_classes, input_shape).create_cnn_model()
    # model.summary()
            
    batch_size = 128
    epochs = 3

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X_tr, tf.transpose(T_train), epochs=epochs, batch_size=batch_size, verbose=0)

    t_hat = model.predict(X_tr).reshape(T_train.shape)
    t_hat_test = model.predict(X_ts).reshape(T_test.shape)

    return compute_nme(t_hat,T_train), compute_nme(t_hat_test,T_test), compute_mse(t_hat,T_train), compute_mse(t_hat_test,T_test), calculate_accuracy(t_hat,T_train), calculate_accuracy(t_hat_test,T_test)