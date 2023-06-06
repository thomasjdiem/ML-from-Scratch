import numpy as np
from util import *

k = 5
debug = True

def Predict(Example):
    distances = np.linalg.norm(Example-X_train,axis=1)
    neighbors_ind = np.argpartition(distances,k)[:k]
    neighbors = [Y_train[ind] for ind in neighbors_ind]
    neighbors_freq = np.bincount(neighbors)
    return np.argmax(neighbors_freq)

def Test():
    CM = np.zeros((10,10),dtype=int)

    #Test on 2.5% of test data since KNN algorithm takes very long to test
    n_train = int(0.025*np.shape(X_test)[0])

    print("Starting testing.")
    for i in range(n_train):                                       
        if i % 100 == 0 and i > 0:
            print(f"Completed testing on example {i} of {n_train}.")
        result = Predict(X_test[i])
        CM[result,Y_test[i]] += 1

    print(f"Completed testing.\nAccuracy: {np.trace(CM)*100/(np.sum(CM)):0.2f}%")
    if debug:
        print("Confusion Matrix: ")
        print(CM)


def main():

    global X_train,Y_train,X_test,Y_test
    X_train,Y_train,X_test,Y_test = LoadMNIST()
    Test()

if __name__ == "__main__":
    main()
