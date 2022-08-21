from sklearn.neural_network import MLPClassifier # Multi Layer Perceptron Classifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

# Load dataset (features and target)
X, y = load_digits(return_X_y=True)
# Split to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

mlp = MLPClassifier()

# Train the model
mlp.fit(X_train, y_train)

# Display first 25 digits in the dataset
fig, axes = plt.subplots(5, 5, figsize=(16, 9))
for i, ax in enumerate(axes.ravel()):
    ax.matshow(X_test[i].reshape(8, 8), cmap=plt.cm.gray)
    ax.set_xticks(())
    ax.set_yticks(())
    num = "predict : " + ''.join(str(x) for x in mlp.predict([X_test[i]]))
    ax.set_title(num)
plt.show()

print("Model Accuracy: ", round(mlp.score(X_test, y_test), 2))