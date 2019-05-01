import numpy as np

# linear kernel
def linearKernel(X,Xt = None):
    if (Xt is None):
        return np.matmul(X,X.T)
    else:
        return np.matmul(X,Xt.T)

def labelledLinearKernel(X,y):
    return linearKernel(X)*np.outer(y,y)



# gaussian/rbf kernel
def rbfKernel(X,sigma=0.01,Xt = None):
    if (Xt is None):
        Q =  np.exp(-sigma*(np.sum((X[None,:] - X[:, None])**2, -1)))
    else:
        Q =  np.exp(-sigma*(np.sum((Xt[None,:] - X[:, None])**2, -1)))
    return Q

def labelledRbfKernel(X,y):
    return rbfKernel(X)*np.outer(y,y)


# tanh kernel
def tanhKernel(X,params=[-1,1],Xt = None):
    if (Xt is None):
        Q =  np.tanh(params[0]*np.matmul(X,X.T)+params[1])
    else:
        Q =  np.tanh(params[0]*np.matmul(X,Xt.T)+params[1])
    return Q

def labelledTanhKernel(X,y):
    return tanhKernel(X)*np.outer(y,y)


# poly kernel
def polyKernel(X,params=[2,1],Xt = None):
    if (Xt is None):
        Q =  (params[1]*np.matmul(X,X.T)+params[2])**params[0]
    else:
        Q =  (params[1]*np.matmul(X,Xt.T)+params[2])**params[0]
    return Q

def labelledPolyKernel(X,y):
    return polyKernel(X)*np.outer(y,y)
