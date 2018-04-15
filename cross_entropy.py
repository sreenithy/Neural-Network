import numpy as np


class CrossEntropyLayer(object):
    def __init__(self):
        """
        Constructor
        """
        self.x = None
        self.t = None

    def forward(self, x, t):
        """
        Implements forward pass of cross entropy

        l(x,t) = 1/N * sum(log(x) * t)

        where
        x = input (number of samples x feature dimension)
        t = target with one hot encoding (number of samples x feature dimension)
        N = number of samples (constant)

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x feature dimension
        t : np.array
            The target data (one-hot) of size number of training samples x feature dimension

        Returns
        -------
        np.array
            The output of the loss

        Stores
        -------
        self.x : np.array
             The input data (need to store for backwards pass)
        self.t : np.array
             The target data (need to store for backwards pass)

        self.t = np.copy(t)
        self.x = np.copy(x)
        N=x.shape[0]
        logv=np.log(x)
        loss=-1*np.sum(logv*t)/N

        """
        self.t = np.copy(t)
        self.x = np.copy(x)

        y = np.sum(t * np.log(x), axis=1)
        #y = np.sum(t * np.log(x + 1e-10), axis=1, keepdims=True)
        y = (-np.sum(y)/self.x.shape[0])
        return (y)


    def backward(self, y_grad=None):
        """
        Compute "backward" computation of softmax loss layer

        Returns
        -------
        np.array
            The gradient at the input

        """
        #Runtime Error
        #return (-1.0/self.x.shape[0])*(self.t/(self.x + 1e-10))
        return (-1.0 / self.x.shape[0]) * (self.t / (self.x ))


