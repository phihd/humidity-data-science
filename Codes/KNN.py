import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X_train_path = 'weather_data_train.csv'
Y_train_path = 'weather_data_train_labels.csv'
X_test_path = 'weather_data_test.csv'
Y_test_path = 'weather_data_test_labels.csv'

# Read dataset to pandas dataframe
X_train = pd.read_csv(X_train_path, index_col='datetime', sep=';', decimal=',', infer_datetime_format=True)
Y_train = pd.read_csv(Y_train_path, index_col='datetime', sep=';', decimal=',', infer_datetime_format=True)
X_test = pd.read_csv(X_test_path, index_col='datetime', sep=';', decimal=',', infer_datetime_format=True)
Y_test = pd.read_csv(Y_test_path, index_col='datetime', sep=';', decimal=',', infer_datetime_format=True)
dataset_train = X_train.merge(Y_train, left_on='datetime', right_on='datetime')
dataset_test = X_test.merge(Y_test, left_on='datetime', right_on='datetime')

train = dataset_train.iloc[:, :-2].values
train_labels = dataset_train.iloc[:, 16].values
test = dataset_test.iloc[:, :-2].values
test_labels = dataset_test.iloc[:, 16].values

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

from sklearn.neighbors import KNeighborsClassifier

error = []
for i in range(1,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train, train_labels)
    pred_i = knn.predict(test)
    error.append(np.mean(pred_i != test_labels))
from sklearn.metrics import classification_report, confusion_matrix

print('Confusion_matrix:')
print(confusion_matrix(test_labels, pred_i))
print()
print(classification_report(test_labels, pred_i))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 50), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()
