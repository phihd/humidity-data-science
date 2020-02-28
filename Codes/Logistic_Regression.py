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


from sklearn.decomposition import PCA

pca = PCA(n_components=0.99)
pca.fit(train)
reduced_train = pca.transform(train)
reduced_test = pca.transform(test)



from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(train, train_labels)
pred = LR.predict(test)

from sklearn.metrics import classification_report, confusion_matrix
print('Confusion_matrix:')
print(confusion_matrix(test_labels, pred))
print()
print(classification_report(test_labels, pred))
