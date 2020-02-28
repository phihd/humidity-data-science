# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 13:55:49 2019

@author: aukus
"""
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing

X_train = pd.read_csv(r'weather_data_train.csv', index_col = 'datetime', sep = ';', decimal = ',', infer_datetime_format=True)
X_train_labels = pd.read_csv(r'weather_data_train_labels.csv', index_col = 'datetime', sep = ';', decimal = ',', infer_datetime_format=True)
X_test  = pd.read_csv(r'weather_data_test.csv', index_col = 'datetime', sep = ';', decimal = ',', infer_datetime_format=True)
X_test_labels =pd.read_csv(r'weather_data_test_labels.csv', index_col = 'datetime', sep = ';', decimal = ',', infer_datetime_format=True)

bins = (-30, 0, 30)
plt.hist(X_train['Tx_mu'], alpha=0.5, label = 'Tx_mu')
plt.hist(X_train['Tn_mu'], alpha=0.5, label ='Tn_mu')
plt.show()
df = X_train.copy()
df['U_mu'] = X_train_labels['U_mu']
df['OBSERVED'] = X_train_labels['OBSERVED']
#sns.pairplot(X_train)
#sns.show()
#print(df.head(3))
#CorrMat = df.corr(method = 'spearman')
#print(CorrMat)
#CorrMat = df.corr(method = 'kendall')
#print(CorrMat)
#CorrMat = df.corr()
#print(CorrMat)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
train = scaler.transform(X_train)
test = scaler.transform(X_test)

pca = PCA(n_components = 2)
pca.fit(train)
print("\nPCA components:\n", pca.components_)

print("\nPCA explained variance:\n", pca.explained_variance_)
print("PCA explain variance ratio:\n", pca.explained_variance_ratio_)

principalcomponents = pca.fit_transform(train)
pca_df = pd.DataFrame(data = principalcomponents, columns = ['PC1', 'PC2'])

print(pca_df.head(5))

#df_ext = df
#df_ext['pc1'] = pca_df['principal component 1']
#df_ext['pc2'] = pca_df['principal component 2']

#print(df_ext.head(5))
#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.show()
Norm = preprocessing.normalize(X_train)

Norm_test = preprocessing.normalize(X_test)

R = linear_model.LinearRegression()
R.fit(Norm, X_train_labels['U_mu'])
print("R coef:\n", R.coef_)
predictions = R.predict(Norm_test)
print(predictions[0:5])
print(R.score(Norm, X_train_labels['U_mu']))
print(R.intercept_)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print("MSE:", mean_squared_error(X_test_labels['U_mu'], predictions))
print("RMSE:", np.sqrt(mean_squared_error(X_test_labels['U_mu'], predictions)))
print("MAE:", mean_absolute_error(X_test_labels['U_mu'], predictions))
print("R square score:", r2_score(X_test_labels['U_mu'], predictions))
predicted_df = pd.DataFrame(predictions)

plt.figure(figsize = (20, 15))
plt.scatter(x = range(X_test_labels['U_mu'].size), y = X_test_labels['U_mu'])
plt.ylabel('Humidity', fontsize=13)
plt.xlabel('Time', fontsize=13)
plt.show()

plt.figure(figsize = (20, 15))
plt.plot(predictions, color = 'r')
plt.ylabel('Relative Humidity', fontsize = 20)
plt.xlabel('Time', fontsize = 20)
plt.title('Regression Model', fontsize = 40)
#plt.scatter(range(predictions.size), X_test_labels['U_mu'].values, color = 'b')
plt.show()

plt.figure(figsize = (20, 15))
plt.plot(predictions, color = 'r')
plt.ylabel('Relative Humidity', fontsize = 20)
plt.xlabel('Time', fontsize = 20)
plt.title('Regression Model', fontsize = 40)
plt.scatter(range(predictions.size), X_test_labels['U_mu'].values, color = 'b')
plt.show()

#Fitting the PCA algorithm with our Data
pca3 = PCA().fit(train)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure(figsize = (20, 15))
plt.plot(np.cumsum(pca3.explained_variance_ratio_))
plt.xlabel('Number of Components', fontsize = 20)
plt.ylabel('Variance (%)', fontsize = 20) #for each component
plt.title('Pulsar Dataset Explained Variance', fontsize = 30)
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
RMSE_list = []
MSE_list = []
r2_list = []
for i in range (1, 17):
    pca3 = PCA(n_components = i)
    pca3.fit(train)
    reduced_train = pca3.transform(train)
    reduced_test = pca3.transform(test)
    #print("After using PCA with", i, "components:\n")
    Norm = preprocessing.normalize(reduced_train)
    Norm_test = preprocessing.normalize(reduced_test)
    R = linear_model.LinearRegression()
    R.fit(Norm, X_train_labels['U_mu'])
    #print("R coef:\n", R.coef_)
    predictions = R.predict(Norm_test)
    #print(predictions[0:5])
    #print(R.score(Norm, X_train_labels['U_mu']))
    #print(R.intercept_)
    #print("MSE:", mean_squared_error(X_test_labels['U_mu'], predictions))
    MSE_list.append(mean_squared_error(X_test_labels['U_mu'], predictions))
    #print("RMSE:", np.sqrt(mean_squared_error(X_test_labels['U_mu'], predictions)))
    RMSE_list.append(np.sqrt(mean_squared_error(X_test_labels['U_mu'], predictions)))
    #print("MAE:", mean_absolute_error(X_test_labels['U_mu'], predictions))
    #print("R square score:", r2_score(X_test_labels['U_mu'], predictions))
    r2_list.append(r2_score(X_test_labels['U_mu'], predictions))

print(MSE_list)
plt.figure(figsize = (20, 15))
plt.plot(MSE_list)
plt.xlabel('Number of components', fontsize = 20)
#plt.xlim(xmin = 0, xmax = 17)
plt.ylabel('MSE', fontsize = 20)
plt.show()


#Polynomial model
from sklearn.preprocessing import PolynomialFeatures
polynomial_features= PolynomialFeatures(degree=2)
X_train_poly = polynomial_features.fit_transform(X_train)
X_test_poly = polynomial_features.fit_transform(X_test)
Norm = preprocessing.normalize(X_train_poly)
Norm_test = preprocessing.normalize(X_test_poly)
R.fit(Norm, X_train_labels['U_mu'])
poly_pred = R.predict(Norm_test)
print("MSE poly:", mean_squared_error(X_test_labels['U_mu'], poly_pred))
print("R2_score poly:", r2_score(X_test_labels['U_mu'], poly_pred))
#tulos = []

#for i in range (0, Norm_test.shape[0]):
#   tulos += Norm_test[i] * R.coef_[i]

#print(tulos.shape)
X_train = pd.read_csv(r'weather_data_train.csv', index_col = 'datetime', sep = ';', decimal = ',', infer_datetime_format=True)
X_train_labels = pd.read_csv(r'weather_data_train_labels.csv', index_col = 'datetime', sep = ';', decimal = ',', infer_datetime_format=True)
X_test  = pd.read_csv(r'weather_data_test.csv', index_col = 'datetime', sep = ';', decimal = ',', infer_datetime_format=True)
X_test_labels =pd.read_csv(r'weather_data_test_labels.csv', index_col = 'datetime', sep = ';', decimal = ',', infer_datetime_format=True)
y_train = X_train_labels['OBSERVED']
y_test = X_test_labels['OBSERVED']
#print(X.shape)
#print(X_test.shape)
#print(X.head(5))

def knn_optimize(self, show_plot=True):
    """
    Finds the optimal minimum number of neighbors to use for the KNN classifier.
    :param show_plot: bool, when True shows the plot of number of neighbors vs error
            Default: False
    :return: the number of neighbors (int)
    """
   
    
    error = []

    
    for i in range(1, 50):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))

    m = min(error)
    min_ind = error.index(m)

    if show_plot:
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, 50), error, color='red', linestyle='dashed', marker='o',
                 markerfacecolor='blue', markersize=10)
        plt.title('Error Rate K Value', fontsize = 40)
        plt.xlabel('K Value', fontsize = 20)
        plt.ylabel('Mean Error', fontsize = 20)
        plt.show()
        

    return min_ind + 1

min_ind = knn_optimize(100, show_plot=True)
print("The method 'knn_optimize' found that " + "the mean error reaches the minimum when the number of neighbors K is", min_ind)
#knn_optimize(100)
knn = KNeighborsClassifier(n_neighbors = min_ind, metric ='euclidean')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
x_pred = knn.predict(X_train)
print("Confusion matrix KNN:\n", confusion_matrix(y_test, y_pred))
print()
print(classification_report(y_test, y_pred))
print(classification_report(y_train, x_pred))

LR = LogisticRegression()
LR.fit(X_train, y_train)
prediction = LR.predict(X_test)
#print("Prediction:\n", prediction)

cm = metrics.confusion_matrix(y_test, prediction)
print("Confusion matrix Logistic:\n", cm)
print()
print(classification_report(y_test, prediction))



def knn_optimize2(self, show_plot=True):
    """
    Finds the optimal minimum number of neighbors to use for the KNN classifier.
    :param show_plot: bool, when True shows the plot of number of neighbors vs error
            Default: False
    :return: the number of neighbors (int)
    """
   
    
    error = []

    
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(reduce_X_train, y_train)
        pred_i = knn.predict(reduce_X_test)
        error.append(np.mean(pred_i != y_test))

    m = min(error)
    min_ind2 = error.index(m)
    print(min_ind2)

    if show_plot:
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
                 markerfacecolor='blue', markersize=10)
        plt.title('Error Rate K Value')
        plt.xlabel('K Value')
        plt.ylabel('Mean Error')
        plt.show()
        

    return min_ind2 + 1

print("After PCA:")

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
train = scaler.transform(X_train)
test = scaler.transform(X_test)

pca10 = PCA(n_components = 16)
pca10.fit(X_train)
reduce_X_train = pca10.transform(train)
reduce_X_test = pca10.transform(test)

min_ind2 = knn_optimize2(100, show_plot=True)
print("The method 'knn_optimize' found that " + "the mean error reaches the minimum when the number of neighbors K is", min_ind2)
#knn_optimize2(100)
knn = KNeighborsClassifier(n_neighbors=36, metric ='euclidean')
knn.fit(reduce_X_train, y_train)
y_pred = knn.predict(reduce_X_test)
x_pred = knn.predict(reduce_X_train)
print(confusion_matrix(y_test, y_pred))
print()
print(classification_report(y_test, y_pred))
print(classification_report(y_train, x_pred))

LR = LogisticRegression()
LR.fit(reduce_X_train, y_train)
prediction = LR.predict(reduce_X_test)
#print("Prediction:\n", prediction)

cm = metrics.confusion_matrix(y_test, prediction)
print("Confusion matrix Logistic:\n", cm)
print()
print(classification_report(y_test, prediction))
