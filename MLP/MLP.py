import numpy as np

__author__ = 'ShuangHe - Student Number: 17210131'

class MLP(object):

    def __init__(self, input_size, hidden_size, output_size):
        '''
        Construct a single hidden layer neural network based on input_size, hidden_size, output_size
        :param input_size: The Number of units in the input layer.
        :param hidden_size: The Number of units in the hidden layer.
        :param output_size: The Number of units in the output layer.
        '''
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.__activation_hidden = None  # Initialize the hidden layer activation function to be empty, that is, linear output.
        self.__activation_output = None  # Initialize the output layer activation function to be empty, that is, linear output.
        self.__loss_func = 'mse'  # Set MSE as loss function by default.

    def __randomise(self):
        '''
        Randomly initialize the parameters of the neural network.
        '''
        self.W1 = np.random.random((self.input_size, self.hidden_size))  # Parameters from the input layer to the hidden layer.
        self.b1 = np.random.random((self.hidden_size,))  # Bias item on the hidden layer.
        self.W2 = np.random.random((self.hidden_size, self.output_size))  # Parameters from the hidden layer to the output layer.
        self.b2 = np.random.random((self.output_size,))  # Bias item on the output layer.

    def __sigmoid(self, x):
        '''
        Return sigmoid(x), the shape of x is (1, size), the return value has the same shape with x.
        '''
        return 1 / (1 + np.exp(-x))

    def __dsigmoid(self, x):
        '''
        Return the derivative of the sigmoid function at x, the shape of x is (1, size),
        and the return value has the same shape as x.
        '''
        return self.__sigmoid(x) * (1 - self.__sigmoid(x))

    def __softmax(self, x):
        '''
        Return softmax(x), the shape of x is (1, size), the return value has the same shape with x.
        '''
        return np.exp(x) / np.sum(np.exp(x))

    def __dsoftmax(self, x):
        '''
        Return the derivative of softmax at x, the shape of x is (1, m),
        and return the m*m matrix. (Vector pair vector derivation)

        '''
        y = self.__softmax(x)  # y: (1, output_size)
        y = np.squeeze(y)  # Reduce dimension y: (output_size, )
        # print('--------softmax(x)-----------')
        # print(y)
        m = x.shape[1]
        prime_mat = np.zeros((m, m))  # Store the derivative matrix.
        for i in range(m):
            for j in range(m):
                if i == j:
                    prime_mat[i, j] = y[i] * (1 - y[i])
                else:
                    prime_mat[i, j] = - y[i] * y[j]
        return prime_mat

    def __activate(self, x, activation_name):
        '''
        The corresponding activation function is implemented according to the activation function name,
        and the activation value is returned.
        '''
        if activation_name is None:
            return x
        elif activation_name == 'sigmoid':
            return self.__sigmoid(x)
        elif activation_name == 'tanh':
            return np.tanh(x)
        elif activation_name == 'softmax':
            return self.__softmax(x)
        else:
            raise Exception("Please assign a valid activation function.")

    def __dactivate(self, x, activation_name):
        '''
        Find the derivative of the activation function at x according to the activation function name.s
        '''
        if activation_name is None:
            return np.ones_like(x)
        elif activation_name == 'sigmoid':
            return self.__dsigmoid(x)
        elif activation_name == 'tanh':
            return 1 - (np.tanh(x))**2
        elif activation_name == 'softmax':
            return self.__dsoftmax(x)
        else:
            raise Exception("Please assign a valid activation function.")


    # np.dot: Matrix multiplication
    # np.inner: inner product
    # *: element wise multiplication

    def forward(self, x):
        '''
        Forward propagation, recording output of each layer
        :param x: shape (1, input_size)
        :return: shape (1, output_size)
        '''

        # Input layer to hidden layer.
        self.Z1 = np.dot(x, self.W1) + self.b1
        self.H = self.__activate(self.Z1, self.__activation_hidden)
        # print('-----------H----------------')
        # print(self.H)

        # Hidden layer to output layer.
        self.Z2 = np.dot(self.H, self.W2) + self.b2
        self.O = self.__activate(self.Z2, self.__activation_output)
        # print('-----------O----------------')
        # print(self.O)

        return self.O

    def backwards(self, x, y):
        '''
        Backpropagation, get the corresponding dW1, db1, dW2, db2 of W1, b1, W2, b2
        :param x: input of the sample
        :param y: label of the sample
        :return:
        '''
        # Determine the back propagation of the error at the output layer based on the loss function
        if self.__loss_func == 'crossEntropy':
            self.loss = np.sum(- y * np.log(self.O))  # record loss
            dO = np.dot(- y / self.O, self.__dactivate(self.Z2, self.__activation_output))
        elif self.__loss_func == 'mse':
            self.loss = np.mean((self.O - y) ** 2) / 2
            dO = (self.O - y) * self.__dactivate(self.Z2, self.__activation_output)
        else:
            raise Exception("Please assign a valid loss function.")

        dW2 = np.dot(self.H.T, dO)
        db2 = dO

        dI = np.dot(dO, self.W2.T) * self.__dactivate(self.Z1, self.__activation_hidden)
        dW1 = np.dot(x.T, dI)
        db1 = dI

        return dW2, db2, dW1, db1

    def update(self, dW2, db2, dW1, db1):
        self.W2 = self.W2 - self.lr * dW2
        self.b2 = self.b2 - self.lr * db2
        self.W1 = self.W1 - self.lr * dW1
        self.b1 = self.b1 - self.lr * db1

    def train(self, x, y, iterations=1000, lr=0.01,
              activation_hidden='tanh', activation_output='tanh', loss_func='crossEntropy'):
        self.lr = lr  # learning rate of gradient descent algorithm
        self.__activation_hidden = activation_hidden  # activation function of the hidden layer
        self.__activation_output = activation_output  # activation function of the output layer
        self.__loss_func = loss_func  # loss function，'mse' by default
        self.losses = []  # Store the loss of every iteration.
        n = x.shape[0]  # sample size

        # Randomly initialize parameters.
        self.__randomise()

        for i in range(iterations):
            loss = 0
            dW2 = np.zeros_like(self.W2)
            db2 = np.zeros_like(self.b2)
            dW1 = np.zeros_like(self.W1)
            db1 = np.zeros_like(self.b1)

            # For a sample of input, conduct forward propagation and back propagation respectively,
            # accumulate the gradient and losses.
            for j in range(n):
                _ = self.forward(x[j, :].reshape((1, -1)))  # forward propagation
                delta = self.backwards(x[j, :].reshape((1, -1)), y[j].reshape((1, -1)))  # back propagation，get the gradient.
                dW2, db2, dW1, db1 = dW2 + delta[0], db2 + delta[1], dW1 + delta[2], db1 + delta[3]
                loss += self.loss

            # After accumulating the gradient of the complete sample, calculate the average gradient,
            # and the loss of this iteration is calculated as the average loss.
            dW2, db2, dW1, db1 = dW2 / n, db2 / n, dW1 / n, db1 / n
            loss /= n
            self.losses.append(loss)  # average loss

            # update parameters based on the average gradient
            self.update(dW2, db2, dW1, db1)

            # if i % 100 == 0:
            #     print('epoch: %d, loss: %-.5f' % (i, loss))
            print('epoch: %d, loss: %-.5f' % (i, loss))

    def predict(self, x):
        '''
        predict input x based on the trained parameters根据训练好的参数，对输入的x进行预测
        :param x: the shape of x is (batch_size, input_size)
        :return:
        '''
        n = x.shape[0]  # sample input size
        y = []  # record output values
        for i in range(n):
            y.append(np.squeeze(self.forward(x[i, :].reshape((1, -1)))))
        y = np.array(y)
        return y

    def getLoss(self):
        '''
        Return the list of losses of each iteration during the training process.
        '''
        return self.losses
