import numpy as np


class FullLayer(object):
    def __init__(self, n_i, n_o):
        """
        Fully connected layer

        Parameters
        ----------
        n_i : integer
            The number of inputs
        n_o : integer
            The number of outputs
        """
        self.x = None
        self.W_grad = None
        self.b_grad = None
        stdev = np.sqrt(np.divide(2.0, n_i + n_o))
        #np.random.seed(100)
        self.W =  np.random.normal(0,stdev,size=(n_o,n_i))
        self.b = np.zeros((1,n_o))

    def forward(self, x):
        """
        Compute "forward" computation of fully connected layer

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
        self.x : np.array
             The input data (need to store for backwards pass)
        """
        #print("full frwd x shape:",x.shape)
        y=np.zeros((x.shape[0],self.W.shape[0]))
        y=np.dot(x,np.transpose(self.W))+self.b
        self.x=np.copy(x)
        return y


    def backward(self, y_grad):
        """
        Compute "backward" computation of fully connected layer

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input

        Stores
        -------
        self.b_grad : np.array
             The gradient with respect to b (same dimensions as self.b)
        self.W_grad : np.array
             The gradient with respect to W (same dimensions as self.W


        """
        self.b_grad = y_grad.sum(axis=0)
        dim = len(self.b_grad)
        self.b_grad = self.b_grad.reshape((1, dim))
        self.W_grad = (np.dot(np.transpose(y_grad), self.x))

        return y_grad.dot(self.W)


    def update_param(self, lr):
        """
        Update the parameters with learning rate lr

        Parameters
        ----------
        lr : floating point
            The learning rate

        Stores
        -------
        self.W : np.array
             The updated value for self.W
        self.b : np.array
             The updated value for self.b
        """


        self.W=self.W-lr*self.W_grad
        self.b = self.b - lr*self.b_grad

