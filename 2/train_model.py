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
    model.fit(x=inp, y=out, batch_size=32, epochs=300, verbose=2, callbacks=None, shuffle=True, initial_epoch=0)
    modelName = "trained.h5"
    model.save(modelName)
    return model

def main():
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    mod = get_model()
    tr_mod = trainModel(mod, x_train, y_train)
main()
