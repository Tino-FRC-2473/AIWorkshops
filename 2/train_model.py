import numpy as np
import keras
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.datasets import boston_housing
from keras.layers import Dense

def get_model():
    model = Sequential()
    model.add(Dense(20, input_dim=13, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1, init='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def trainModel(model, inp, out):
    tbCallBack = keras.callbacks.TensorBoard(log_dir='logger/run_a', histogram_freq=0.0016, batch_size=32, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    model.fit(x=inp, y=out, batch_size=32, epochs=300, verbose=2, callbacks=[tbCallBack], shuffle=True, initial_epoch=0)
    modelName = "trained.h5"
    model.save(modelName)
    return model

def main():
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    mod = get_model()
    tr_mod = trainModel(mod, x_train, y_train)
main()
