import numpy as np
import matplotlib.pyplot as plt
import utils.kernels as KN
import time
from mpl_toolkits import mplot3d




# data generators
#================
def largerBinaryProblem(N=1000,D=20,prop=1):
    # data generation
    mean = np.ones(D)
    std = np.random.randint(2,size=D)+1
    x0 = np.random.normal(loc=mean, scale=std, size=(N, D))
    x0t = np.random.normal(loc=mean, scale=std, size=(prop*N, D))

    mean = -np.ones(D)
    std = np.random.randint(2,size=D)+1
    x1 = np.random.normal(loc=mean, scale=std, size=(N, D))
    x1t = np.random.normal(loc=mean, scale=std, size=(prop*N, D))

    X = np.concatenate((x0,x1));
    y = np.concatenate((np.ones((N,1)),-np.ones((N,1)))).T
    Xt = np.concatenate((x0t,x1t));
    yt = np.concatenate((np.ones((prop*N,1)),-np.ones((prop*N,1)))).T
    return X,y[0],Xt,yt[0]

def largerRegressionProblem(N=1000,D=20,prop=1):

    # data generation
    X = 3*np.random.random_sample((N,D))
    Xt = 3*np.random.random_sample((prop*N,D))
    y = np.cos(0.2*np.mean(X,axis=1))-0.4*np.sin(-np.std(X,axis=1))+np.sin(-5*X[:,0]*X[:,1])
    yt = np.cos(0.2*np.mean(Xt,axis=1))-0.4*np.sin(-np.std(Xt,axis=1))+np.sin(-5*Xt[:,0]*Xt[:,1])

    return X,y,Xt,yt

def smallBinaryProblem(N=100,prop=1):
    D = 2;
    # data generation
    mean1, mean2, std1, std2 = 1, 3, 2, 2
    x0 = np.random.normal(loc=[mean1, mean2], scale=[std1, std2], size=(N, D))
    x0t = np.random.normal(loc=[mean1, mean2], scale=[std1, std2], size=(prop*N, D))

    mean3, mean4, std3, std4 = 2, -2, 2, 1
    x1 = np.random.normal(loc=[mean3, mean4], scale=[std3, std4], size=(N, D))
    x1t = np.random.normal(loc=[mean3, mean4], scale=[std3, std4], size=(prop*N, D))
    mean3, mean4, std3, std4 = -2, -2, 0.5, 2
    x2 = np.random.normal(loc=[mean3, mean4], scale=[std3, std4], size=(N, D))
    x2t = np.random.normal(loc=[mean3, mean4], scale=[std3, std4], size=(prop*N, D))

    X = np.concatenate((x0,x1,x2));
    Xt = np.concatenate((x0t,x1t,x2t));
    y = np.concatenate((np.ones((N,1)),-np.ones((2*N,1)))).T
    yt = np.concatenate((np.ones((prop*N,1)),-np.ones((2*prop*N,1)))).T
    return X,y[0],Xt,yt[0]




# 2D data plotters
#================
def plot2dResult(X,y,alpha,b,type,C,param = None, num=1,origAlpha = None):
    plt.figure(num=num, figsize=(8,6), dpi=150, facecolor='w', edgecolor='k')
    plt.clf()
    plt.scatter(X[np.where(y==1),0],X[np.where(y==1),1],s=4,c='r')
    plt.scatter(X[np.where(y==-1),0],X[np.where(y==-1),1],s=4,c='b')
    scale = 6.5
    axes = [-scale,scale,-scale,scale]
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    Xt = np.c_[x0.ravel(), x1.ravel()]
    if (type=='linear'):
        K = KN.linearKernel(X,Xt)
    elif (type=='rbf'):
        K = KN.rbfKernel(X,sigma=param,Xt = Xt)
    elif (type=='tanh'):
        K = KN.tanhKernel(X,params=param,Xt = Xt)
    elif (type=='poly'):
        K = KN.polyKernel(X,params=param,Xt = Xt)
    y_pred = (np.dot(y*alpha.T,K)+b).reshape(x0.shape)
    plt.contourf(x0, x1, np.sign(y_pred) ,alpha=0.2)
    plt.contour(x0, x1, y_pred ,[-1,0,1])
    if (origAlpha is None):
        plt.scatter(X[np.where(np.abs(alpha)>0), 0], X[np.where(np.abs(alpha)>0), 1],s=15,edgecolors='y',facecolors='none')
        plt.scatter(X[np.where(np.abs(alpha)==C), 0], X[np.where(np.abs(alpha)==C), 1],s=15,edgecolors='m',facecolors='none')
    else:
        plt.scatter(X[np.where(np.abs(origAlpha)>0), 0], X[np.where(np.abs(origAlpha)>0), 1],s=15,edgecolors='y',facecolors='none')
        plt.scatter(X[np.where(np.abs(origAlpha)==C), 0], X[np.where(np.abs(origAlpha)==C), 1],s=15,edgecolors='m',facecolors='none')
    plt.draw()
    plt.pause(1e-17)
    time.sleep(0.1)


def plot2dResultSVR(X,y,alpha,b,type,C, epsilon, param = None, num=1,origAlpha = None):
    plt.figure(num=num, figsize=(8,6), dpi=150, facecolor='w', edgecolor='k')
    plt.clf()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[:,0],X[:,1],y,c=y)
    scale = 3
    axes = [0,scale,0,scale]
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    Xt = np.c_[x0.ravel(), x1.ravel()]
    if (type=='linear'):
        K = KN.linearKernel(X,Xt)
    elif (type=='rbf'):
        K = KN.rbfKernel(X,sigma=param,Xt = Xt)
    elif (type=='tanh'):
        K = KN.tanhKernel(X,params=param,Xt = Xt)
    elif (type=='poly'):
        K = KN.polyKernel(X,params=param,Xt = Xt)
    y_pred = (np.dot(alpha.T,K)+b).reshape(x0.shape)
    ax.contour3D(x0, x1, y_pred , 100, cmap='plasma')
    plt.draw()
    plt.pause(1e-17)
    time.sleep(0.1)

def plot2dData(X,y):
    plt.scatter(X[np.where(y==1),0],X[np.where(y==1),1],s=20,c='r')
    plt.scatter(X[np.where(y==-1),0],X[np.where(y==-1),1],s=20,c='b')
    plt.show(block=False)

def plot2dDataReg(X,y):
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[:,0],X[:,1],y,c=y)
    plt.show(block=False)
