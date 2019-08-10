import numpy as np
from MLP import MLP
import matplotlib.pyplot as plt

np.random.seed(2)

# one-hot
'''
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
NN = MLP(2, 2, 2)
NN.train(X, y, lr=0.5, iterations=5000, activation_hidden='sigmoid', activation_output='softmax', loss_func='crossEntropy')
y_pred = NN.predict(X)
print(y_pred)

'''
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
# Train an MLP with 2 inputs, two hidden units and one output
NN = MLP(2, 2, 1)
# NN.train(X, y, lr=0.5, iterations=5000, activation_hidden='tanh', activation_output='sigmoid', loss_func='mse')
# NN.train(X, y, lr=0.5, iterations=2000, activation_hidden='tanh', activation_output='tanh', loss_func='mse')
NN.train(X, y, lr=0.5, iterations=5000, activation_hidden='sigmoid', activation_output='tanh', loss_func='mse')
# NN.train(X, y, lr=0.5, iterations=5000, activation_hidden='sigmoid', activation_output='sigmoid', loss_func='mse')  # a very bad result
y_pred = NN.predict(X)
print(y_pred)


# Plot the change in loss during training
losses = NN.getLoss()
plt.figure()
plt.plot(losses)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()


