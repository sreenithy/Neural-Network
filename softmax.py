import numpy as np


class SoftMaxLayer(object):
    def __init__(self):
        """
        Constructor
        """
        self.y = None

    def forward(self, x):
        """
        Implement forward pass of softmax

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number of features

        Returns
        -------
        np.array
            The output of the layer

        Stores
        -------
        self.y : np.array
             The output of the layer (needed for backpropagation)
        """
        xmax=x.max(1).reshape(x.shape[0],1)
        x=x-xmax
        xexp=np.exp(x)
        xsum=xexp.sum(axis=1).reshape(x.shape[0],1)
        self.y=xexp/xsum
        #print("Softmax-forward:",self.y.shape)
        return self.y

    def backward(self, y_grad):
        """
        Compute "backward" computation of softmax

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input

        """
        dlx=np.zeros([y_grad.shape[0],y_grad.shape[1]])

        #x_grad=np.zeros(y_grad.shape)
        for i in range(0,self.y.shape[0]):
            x=np.diag(self.y[i])
            #z=self.y[i].reshape((1,y_grad.shape[1]))

            dyx=x-np.outer(self.y[i].T,self.y[i])
            dlx[i]=np.dot(y_grad[i],dyx).reshape((1,self.y.shape[1]))
            #x_grad[i]=y_grad[i].reshape((1,y_grad.shape[1])).dot(dydx)
        return dlx

    def update_param(self, lr):
        pass  # no learning for softmax layer
