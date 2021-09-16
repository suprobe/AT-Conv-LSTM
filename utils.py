import numpy as np
from keras import backend as K


def rmse_train(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def my_loss(y_true, y_pred):
    mse = K.mean(K.square(y_true - y_pred), axis = -1)
    return  mse

def predict_point_by_point(model, data):
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size, ))
    print ("predict_size:", predicted.size)
    return predicted

def MAE(pre,true):
    c = 0
    b = abs(np.subtract(pre, true))
    for i in range(len(b)):
        c = c + b[i]
    return c / len(b)

def MAPE(pre, true):
    a=0
    for i in range(len(pre)):
        x = (abs(pre[i] - true[i])/true[i])
        a = a + x
    return a / len(pre)

def RMSE(pre,true):
    c = 0
    b = abs(np.subtract(pre, true))
    b = b*b
    for i in range(len(b)):
        c = c + b[i]
    d = (c / len(b))**0.5
    return d