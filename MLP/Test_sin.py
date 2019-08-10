import numpy as np
from MLP import MLP
import matplotlib.pyplot as plt

np.random.seed(2)
X = - 1 + 2 * np.random.random((200, 4))  # Generate 200 samples, each with 4 features, each with a value between -1 and 1
y = np.sin(X[:, 0] - X[:, 1] + X[:, 2] - X[:, 3])  # y = sin(x1 - x2 + x3 - x4)

NN = MLP(4, 16, 1)
# NN.train(X, y, lr=0.1, iterations=1000, activation_hidden='tanh', loss_func='mse')
NN.train(X[:150, :], y[:150], lr=0.1, iterations=1000, activation_hidden='sigmoid', activation_output='tanh', loss_func='mse')

y_train_pred = NN.predict(X[:150, :])  # the shape of y_train_pred: (150,)  training set prediction
y_test_pred = NN.predict(X[150:, :])  # test set prediction


# Calculate the average error on the training set and test set.
i = 0
j = 0
error_train = 0.
error_test = 0.
for i in range(0, 149):
    error_train += 0.5 * (y[:150][i] - y_train_pred[i]) ** 2
print('error on train set : %.5f' % (error_train/150))

for j in range(0, 49):
    error_test += 0.5 * (y[150:][j] - y_test_pred)[j] ** 2
print('error on test set : %.5f' % (error_test/50))



# Plot the comparison between training set predictions and the real situation.
plt.figure(0)
plt.plot(y_train_pred, label='y_pred')
plt.plot(y[:150], label='y_true')
plt.title("training")


# Plot the comparison between test set predictions and the real situation.
plt.figure(1)
plt.plot(y_test_pred, label='y_pred')
plt.plot(y[150:], label='y_true')
plt.title("testing")
plt.legend()

# Plot the change in loss during the training process.
losses = NN.getLoss()
plt.figure(2)
plt.plot(losses)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()


