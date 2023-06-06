import numpy as np
from util import *

max_iter = 100000
kernel = lambda X1,X2: PolynomialKernel(X1,X2,2)
std_init_weights = 0.3
epsilon = 0.001
    
def Predict(X,w,b):
    return np.sign(np.dot(w.T, X.T) + b).astype(int)

def Fit(X,y):
    n_samples = X.shape[0]

    alpha = np.random.normal(0,std_init_weights,size=n_samples)

    iteration = 0 
    while True:
        if iteration == max_iter:
            print(f"Parameters did not converge after {max_iter} iterations.")
            return

        alpha_prev = np.copy(alpha)
        for i in range(n_samples):
            j = np.random.choice(range(n_samples))
            K = kernel(X[i,:],X[i,:]) + kernel(X[j,:],X[j,:]) - 2*kernel(X[i,:],X[j,:])

            if K == 0:
                continue

            if y[i] == y[j]:
                L = max(0, alpha[i] + alpha[j] - 1) #C = 1
                H = min(1, alpha[i] + alpha[j]) # C =1
            else:
                L = max(0, alpha[i] - alpha[j])
                H = min(1, 1 - alpha[i] + alpha[j]) #C=1

            w = alpha*y @ X
            b = np.mean(y-w.T@X.T)

            yhat = Predict(X[i,:],w,b)
            Loss_i = yhat - y[i]

            yhat = Predict(X[j,:],w,b)
            Loss_j = yhat - y[j]  

            alpha_i_old = alpha[i]
            alpha[i] += float(y[i] * (Loss_j - Loss_i))/K
            alpha[i] = max(alpha[i], L)
            alpha[i] = min(alpha[i], H)  

            alpha[j] += y[i]*y[j] * (alpha_i_old - alpha[i])

        change_norm = np.linalg.norm(alpha - alpha_prev)
        if change_norm < epsilon:
            print(f"Parameters converged after {iteration} iterations.")
            break
        elif change_norm  == float("inf"):
            print(f"Parameters diverged after {iteration} iterations.")
            break

        iteration += 1

y = []
X = []
with open("bezdekIris.data","r") as f:
    for line in f:
        values = line.split(',')
        flower = values[-1]
        if flower[:14] == "Iris-virginica":
            y.append(1)
        else:
            y.append(-1)
        X.append([float(value) for value in values[:-1]])


X.pop(-1)
y.pop(-1)
X = np.array(X)
y = np.array(y)

X_train, y_train = X[:110], y[:110]
X_test, y_test = X[110:], y[110:]

Fit(X_train,y_train)



