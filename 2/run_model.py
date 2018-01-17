import numpy as np
import keras
from keras.models import load_model
from keras.datasets import boston_housing

def run_model(model, inp):
    inp_test = np.array(inp).reshape((1,) + np.array(inp).shape)
    pred = model.predict(inp_test)
    return pred[0][0]

def main():
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    model = keras.models.load_model("trained.h5")
    print(x_test[65])
    print "actual: "+str(y_test[65])
    print "predicted: "+str(run_model(model, x_test[65]))

main()
