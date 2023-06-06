import numpy as np
from util import *
       
alpha = 0.05
NN_Structure = [784,400,200,10]
ActivationFunctions = [LinearActivation,SigmoidActivation,TanhActivation,SoftMaxActivation]
num_epochs = 3
std_init_weights = 0.1
debug = True

def Initialize_NN():
    global N_layers,Weights,Biases,Neurons,Neurons_Activation
    N_layers = len(NN_Structure)
    assert len(ActivationFunctions) == N_layers
    Weights = []
    Biases = []
    Neurons = [0 for _ in range(N_layers)]
    Neurons_Activation = [0 for _ in range(N_layers)]
    for i in range(N_layers-1):
        Weights.append(np.random.normal(0,std_init_weights,size=(NN_Structure[i],NN_Structure[i+1])))
        Biases.append(np.random.normal(0,std_init_weights,size=(1,NN_Structure[i+1])))

    print(f"Successfully initialized {'-'.join([str(i) for i in NN_Structure])} Neural Network.")


def ForwardPropagate(Input):
    Neurons[0] = np.array([Input])
    Neurons_Activation[0] = ActivationFunctions[0].f(np.array([Input]))
    for i in range(N_layers-1):
        Neurons[i+1] = Neurons_Activation[i] @ Weights[i] + Biases[i]
        Neurons_Activation[i+1] = ActivationFunctions[i+1].f(Neurons[i+1])

def BackPropagate(Target):

    grad = Neurons_Activation[-1]-Target
    grad *= 0.1

    Weights[-1] -= alpha * np.transpose(Neurons_Activation[-2]) @ grad
    Biases[-1] -= alpha * grad

    for i in range(N_layers-2,0,-1):
        grad = grad @ np.transpose(Weights[i])
        grad *= ActivationFunctions[i].df(Neurons[i])

        Weights[i-1] -= alpha*np.transpose(Neurons_Activation[i-1]) @ grad
        Biases[i-1] -= alpha*grad


def Train(X_train,Y_train):
    for epoch in range(num_epochs):
        print(f"Started training on epoch {epoch+1} of {num_epochs}.")
        for i in range(np.shape(X_train)[0]):
            ForwardPropagate(X_train[i,:])
            Target = np.zeros(10)
            Target[Y_train[i]] = 1
            BackPropagate(Target)

def Test(X_test,Y_test):
    print("Started testing on unseen data.")
    
    CM = np.zeros((10,10),dtype=int)
    for i in range(np.shape(X_test)[0]):
        ForwardPropagate(X_test[i,:])
        result = np.argmax(Neurons_Activation[-1])
        CM[Y_test[i],result] += 1

    print(f"Accuracy: {np.trace(CM)*100/(np.sum(CM)):0.2f}%")
    if debug:
        print("Confusion Matrix: ")
        print(CM)

def main():

    X_train,Y_train,X_test,Y_test = LoadMNIST()
    Initialize_NN()
    Train(X_train,Y_train)
    Test(X_test,Y_test)

if __name__ == "__main__":
    main()