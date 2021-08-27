import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

np.random.seed(12)
B = 10000

# Read in the data
Q1_data = pd.read_csv("Q1.csv") 
# Training data set (n=500)
Q1train = Q1_data.iloc[:500,:].to_numpy()

# Non-parametric bootstrap implementation 
# Generate 10,000 bootstrap samples.
i = 0 
B_sample = {}
regr_coefs = np.zeros((B,45)) # 10,000 rows with 45 features in each row 

while (i < B):
    B_sample[i] = Q1train[np.random.choice(Q1train.shape[0], size = 500, replace = True)]
    Xtrain = B_sample[i][:,0:45]
    Ytrain = B_sample[i][:,-1]

    # Logistic model on B[i] 
    regr = LogisticRegression(C = 1, solver = 'liblinear', penalty = 'l1').fit(Xtrain, Ytrain)
    regr_coefs[i] = regr.coef_[0]
    i = i + 1

# Compute the Bootstrap mean for each feature + lower and upper bounds of CI
B_mean_coefs = np.zeros(45)
regr_coefs_transpose = regr_coefs.transpose() #1 row for each feature with 10,000 elements in each row 
B_Lower = np.zeros(45)
B_Upper = np.zeros(45)
j = 0
while (j < 45):
    i = 0
    while (i < B):
        B_mean_coefs[j] = B_mean_coefs[j] + regr_coefs[i][j]
        i = i + 1
    B_mean_coefs[j] = (1/B) * B_mean_coefs[j]
    B_Lower[j] = np.percentile(regr_coefs_transpose[j], q = 5)
    B_Upper[j] = np.percentile(regr_coefs_transpose[j], q = 95)
    j = j + 1

# Plotting the Confidence Intervals for each feature 
i = 0
label_x = np.empty(45, dtype = object)
while (i < 45):
    label_x[i] = '\u03B2' + str(i + 1)
    i = i + 1
ax = plt.subplot()
plt.xlim([-1,45])
plt.ylim([-2.5,2.5])
ax.set_xticks(np.arange(45))
ax.set_xticklabels(label_x)
ax.axes.set_title("90% Bootstrap Confidence Intervals for Betas", fontsize=16)
plt.xticks(rotation=45)
plt.xticks(fontsize=6)

j = 0
while (j < 45):
    point1 = [j, B_Upper[j]]
    point2 = [j, B_Lower[j]]
    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]

    plt.plot(j, B_mean_coefs[j], marker="o", markersize=4, markeredgecolor="green")

    if (B_Upper[j]*B_Lower[j] <= 0):
        plt.plot(x_values, y_values, color = "red")
    else:
        plt.plot(x_values, y_values, color = "blue")
    j = j + 1

plt.show()

