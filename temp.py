import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import scipy.io as sio
"""
 ---- Loading data ----
"""

file = pd.read_excel("C:\\Users\\Asus\\Desktop\\DA\\Project\\temp.xlsx",header=None)
data = file.to_numpy()
inputs = np.array(data[101-1:115, 3-1:8],dtype=('float64'))
outputs = np.array(data[101-1:115, 11],dtype=('float64'))

X = inputs
Y = outputs

n0 = 6 # input layer
n1 = 6 # hidden layer
n2 = 1 # output layer


epochs=1000


def cm(g,lr):
    def activation(x):
        y = 1/(1 + np.exp(-g * x))
        return y
    def feedforward(input_net):
        x1 = np.dot(input_net , w1.T)
        y1 = activation(x1)
        x2 = np.dot(y1 , w2.T)
        y2 = activation(x2)
        return y1 , y2
    def d_activation(out):
        d_y = out * ( 1 - out)
        return d_y
    w1 = np.load('w1.npy')
    w2 = np.load('w2.npy')
    List_Mse = [] 
    for i in range(epochs):
        for j in range(len(X)):
            input = X[j] # shape input = (n0,)
            input = np.reshape(input , newshape=(1,n0)) # shape input = (1,n0)
            target = Y[j]
            y1 , y2 = feedforward(input)
            error = target - y2
            d_f2 = d_activation(y2)
            diag_d_f2 = np.diagflat(d_f2)
    
            d_f1 = d_activation(y1)
            diag_d_f1 = np.diagflat(d_f1)
    
            temp1 = -2 * error * d_f2
            temp2 = np.dot(temp1 , w2)
            temp3 = np.dot(temp2 , diag_d_f1)
            temp4 = temp3.T
            temp5 = np.dot(temp4, input)
            w1 = w1 - lr * temp5
            w2 = w2 - lr * np.dot(temp1.T , y1)
    
        #calculating MSE and accuracy for Train 
        Netoutput = []
        Target = []
        rnd_Netoutput = []
        for idx in range(len(X)):
            input= X[idx]
            target = Y[idx]
            Target.append(target)
            _ , pred = feedforward(input)
            Netoutput.append(pred)
            rnd_Netoutput.append(np.round(pred))
        
        MSE = mse(Target , Netoutput)
        List_Mse.append(MSE)
        # print('epoch: ' , i , ' , MSE = '  , MSE )
    return List_Mse[-1]
g=[]
for y in range(1,60):
    y /= 10.0
    g.append(y)
lr=[]
for x in range(1,700):
    x /= 100.0
    lr.append(x)
errors=np.zeros((len(g), len(lr)))
for i in range(len(g)):
    g0=g[i]
    for j in range(len(lr)):
        lr0=lr[j]
        errors[i,j]=cm(g0, lr0)
        print('Cell ( ',i, ' , ',j,' )', '\tMSE =   ', cm(g0, lr0))
sio.savemat('mse.mat', {'errors': errors})
