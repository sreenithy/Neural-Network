from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt


class Sequential(object):
    def __init__(self, layers, loss):
        """
        Sequential model

        Implements a sequence of layers

        Parameters
        ----------
        layers : list of layer objects
        loss : loss object
        """
        self.layers = layers
        self.loss = loss

    def forward(self, x, target=None):
        """
        Forward pass through all layers
        
        if target is not none, then also do loss layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number of features
        target : np.array
            The target data of size number of training samples x number of features (one-hot)

        Returns
        -------
        np.array
            The output of the model
        """
        """
        y1 = self.layers[0].forward(x) #full1
        relu = self.layers[1].forward(y1) #relu
        y2 = self.layers[2].forward(relu) #full2
        soft = self.layers[3].forward(y2) #softmax
        if target is None:
            return soft

        else:
            loss = self.loss.forward(soft, target)
            return loss

        
        """
        y1=[]
        no_layers=len(self.layers)
        for i in range(0,no_layers):
            y1=self.layers[i].forward(x)
            print("Calling forward of Layerv", i, "with input shape", x.shape, "and output shape", y1.shape)
            x=np.copy(y1)

        if target is None:
            return y1

        else:
            loss = self.loss.forward(y1, target)
            print("returned loss",loss)
            return loss




    def backward(self):
        """
        Compute "backward" computation of fully connected layer

        Returns
        -------
        np.array
            The gradient at the input

        """

        g1=self.loss.backward()
        for i in self.layers[::-1]:
            g1=i.backward(g1)
        return g1


    def update_param(self, lr):
        """
        Update the parameters with learning rate lr

        Parameters
        ----------
        lr : floating point
            The learning rate
        """
        no_layers = len(self.layers)
        for i in range(0, no_layers):
            self.layers[i].update_param(lr)


    def fit(self, x, y, epochs, lr, batch_size=128):
        """
        Fit parameters of all layers using batches

        Parameters
        ----------
        x : numpy matrix
            Training data (number of samples x number of features)
        y : numpy matrix
            Training labels (number of samples x number of features) (one-hot)
        epochs: integer
            Number of epochs to run (1 epoch = 1 pass through entire data)
        lr: float
            Learning rate
        batch_size: integer
            Number of data samples per batch of gradient descent


        """

        t = x.shape[0] * 1.0 / batch_size

        avgloss = np.zeros((epochs, 1))
        loss = np.zeros((epochs, int(t)))
        for i in range(0, epochs):  # epochs
            for j in range(0, int(t)):  # t
                loss[i, j] = self.forward(x[j * batch_size:((j + 1) * batch_size), :],
                                          y[j * batch_size:((j + 1) * batch_size), :])
                grad = self.backward()
                self.update_param(lr)
            print("Iteration", i, loss[i])
            avgloss[i] = np.sum(loss[i, :]) / t
        #print('Average loss for lf ',lr, 'is' ,avgloss)
        #plt.plot(avgloss)
        #plt.plot(avgloss, 'r-', label='Learning Rate 0.1')
        #plt.show()  # Plot the image



    def predict(self, x):
        """
        Return class prediction with input x

        Parameters
        ----------
        x : numpy matrix
            Testing data data (number of samples x number of features)

        Returns
        -------
        np.array
            The output of the model (integer class predictions)
        """

        y=self.forward(x)
        #print('Predict y',y)
        outputval = np.zeros((y.shape[0], 1))
        for i in range(0,y.shape[0]):

            result=np.amax(y[i], axis=0)

            for j in range(0,y.shape[1]):

                if y[i,j] == result:

                    outputval[i]=j
                #print("result", result,"outval row",outputval[i])
        return outputval

