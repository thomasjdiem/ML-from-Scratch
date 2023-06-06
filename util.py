import pandas as pd
import numpy as np

#Neural Network activation functions
class LinearActivation:
    def f(array):
        return array
    def df(array):
        return np.ones(np.shape(array))

class ReLUActivation:
    def f(array):
        return np.maximum(0,array)
    def df(array):
        return np.greater(array, 0).astype(int)
    
class SigmoidActivation:
    def f(array):
        array = np.clip(array, -500, 500)
        return np.exp(array)/(1+np.exp(array))
    def df(array):
        a = SigmoidActivation.f(array)
        return a*(1-a)
    
class SoftMaxActivation:
    def f(array):
        array = np.clip(array, -500, 500)
        earray = np.exp(array)
        tot = np.sum(earray)
        return earray/tot
    def df(array):
        a = SoftMaxActivation.f(array)
        return a*(1-a)
    
class TanhActivation:
    def f(array):
        return np.tanh(array)
    def df(array):
        return 1 - (np.tanh(array))**2
    
#Support Vector Machine Kernels
def LinearKernel(X1,X2):
    return X1 @ X2.T

def PolynomialKernel(X1,X2,deg):
    return(X1 @ X2.T + 1)**deg

def RBFKernel(X1,X2,gamma):
    return np.exp(-gamma * np.linalg.norm(X1 - X2)**2)

def SigmoidKernel(X1,X2,Beta0,Beta1):
    return np.tanh(Beta0 * X1 @ X2.T + Beta1)

#Load MNIST data set
def LoadMNIST():

    df = pd.read_csv("mnist_train.csv")
    Y_train = df['label'].to_numpy()
    X_train = df.loc[:, df.columns != 'label'].to_numpy() #

    X_train = (X_train)/255  #

    df = pd.read_csv("mnist_test.csv")
    Y_test = df['label'].to_numpy()
    X_test = df.loc[:, df.columns != 'label'].to_numpy()  #

    X_test = (X_test)/255    #

    return X_train,Y_train,X_test,Y_test


    