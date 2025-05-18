import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Load dataset
iris = datasets.load_iris()

# Convert to DataFrame
df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_iris['target'] = iris.target

# Split dataset
data_train, data_test, target_train, target_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=0
)

# Define Neural Network model
clf = MLPClassifier(hidden_layer_sizes=10, activation='relu', solver='adam', max_iter=1000)

# Train model
clf.fit(data_train, target_train)

# Accuracy scores
train_acc = clf.score(data_train, target_train)
print(f"Train Accuracy: {train_acc:.2f}")

# Predictions
predictions = clf.predict(data_test)
print(f'result:{predictions}')
print(f'answer:{target_test}')

# Show loss curve
plt.plot(clf.loss_curve_)
plt.title('Loss Curve')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid()
plt.show()

# Show Confusion Matrix
conf_matrix = confusion_matrix(target_test, predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()