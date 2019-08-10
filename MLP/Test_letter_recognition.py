import numpy as np
from MLP import MLP
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from utils import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler

np.random.seed(2)

# hyper-params
input_size = 16
hidden_size = 32
output_size = 26
lr = 0.1

# Reading data & processing data into a standardized format of input and output
# Each line is a string, the format is like: 'D,2,4,3,3,2,8,7,5,6,9,5,5,2,8,3,7\n'
# The first one puts the label, and the next 16 integers represent 16 features.
X = []
y = []
with open('letter-recognition.data', 'r') as f:
    line = f.readline()  # Read the first line.
    while line:
        # 'D,2,4,3,3,2,8,7,5,6,9,5,5,2,8,3,7\n' -->
        # ['D','2','4','3','3','2','8','7','5','6','9','5','5','2','8','3','7']
        line = line.rstrip('\n').split(',')  # Delete the newline character, and split by a comma.
        # Converts elements of the line except for the first element to an int type
        # and the resulting list is stored in X as features of the sample.
        X.append(list(map(int, line[1:])))
        label = np.zeros((output_size,))  # initialize one-hot labels of the sample
        label[ord(line[0])-ord('A')] = 1  # one-hot encoding
        y.append(label)
        line = f.readline()  # continue reading the next line
X = np.array(X)
y = np.array(y)
# print(X.shape, y.shape)  # >>> (20000, 16) (20000, 26)

# data pre-processing: normalized
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an MLP.
NN = MLP(input_size, hidden_size, output_size)

# training
NN.train(X_train, y_train, lr=lr, iterations=2000, activation_hidden='sigmoid', activation_output='softmax', loss_func='crossEntropy')

# predict
y_train_pred = np.argmax(NN.predict(X_train), axis=1)  # Retrieve the index corresponding to the maximum value in the softmax result.
y_test_pred = np.argmax(NN.predict(X_test), axis=1)
y_train_true = np.argmax(y_train, axis=1)  # One-hot converts to an ordered tag, such as [1, 0, 0... 0] -> 0 indicates label 'A'
y_test_true = np.argmax(y_test, axis=1)

# evaluate
acc_train = accuracy_score(y_train_true, y_train_pred)
acc_test = accuracy_score(y_test_true, y_test_pred)
print("Training accuracy: %.2f%%, Testing accuracy: %.2f%%" % (acc_train * 100, acc_test * 100))
class_names = np.array([chr(i) for i in range(65, 91)])
# print(class_names)
# Plot non-normalized confusion matrix
np.set_printoptions(precision=2)
plot_confusion_matrix(y_train_true, y_train_pred, classes=class_names,
                      title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
plot_confusion_matrix(y_test_true, y_test_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.show()

# Plot the change in loss during training
losses = NN.getLoss()
plt.figure()
plt.plot(losses)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()
