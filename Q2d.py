import pandas as pd
import numpy as np
from numpy import asarray
from sklearn.preprocessing import MinMaxScaler

# 0. read in the data 
Q2_data = pd.read_csv("Q2.csv")

# 1. remove any rows with missing values (NA)
Q2_data = Q2_data.dropna()

# 2. delete all features except for age, nearestMRT and nConvenience
X_Q2_data = Q2_data[["age", "nearestMRT", "nConvenience"]]
Y_Q2_data = Q2_data[["price"]]

# 3. use the sklearn minmaxscaler to normalize the features
scaler = MinMaxScaler()
X_Q2_data = asarray(scaler.fit_transform(X_Q2_data))

# 4. create a training set from the first half of the resulting dataset, and a test set from the remaining half.
Xtrain = asarray(X_Q2_data[:204,:])
Ytrain = asarray(Y_Q2_data.head(204))
Xtest = asarray(X_Q2_data[204:,:])
Ytest = asarray(Y_Q2_data.tail(204))

print("X train first row:", Xtrain[0])
print("X train last row:", Xtrain[203])
print("X test first row:", Xtest[0])
print("X test last row:", Xtest[203])
print("Y train first row:", Ytrain[0])
print("Y train last row:", Ytrain[203])
print("Y test first row:", Ytest[0])
print("Y test last row:", Ytest[203])