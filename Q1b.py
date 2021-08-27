import copy  
import numpy as np 
import pandas as pd
import seaborn as sns
import statistics as s
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

# Read in the data
Q1_data = pd.read_csv("Q1.csv") 

# Split data into Train (first 500 obs) vs Test sets 
train_set = Q1_data.iloc[:500,:]
test_set = Q1_data.iloc[500:,:]

# Grid of 100 C values from {0.0001,...,0.6} inclusive  
param_grid = {'C': list(np.around(np.linspace(0.0001, 0.6, 100),6))}


# split the train data into 10 groups of size 50
NumFolds = 10
Fold_i = {} # collection of data frames 

i = 0 
while (i < NumFolds):
    Fold_i[i] = train_set.iloc[(50 * i):(50 * (i + 1)),:]
    length_fold = len(Fold_i[i])
    i = i + 1


# cross validation implementation 
c_iter = 0 
cv_score = np.zeros((100,10))
cv_avg = np.zeros(100)

for c_value in param_grid['C']:
    j = 0
    while (j < NumFolds):
        x_test = Fold_i[j].iloc[:,0:45]
        y_test = Fold_i[j].iloc[:,-1] 
        
        train_group = copy.deepcopy(Fold_i)
        train_group.pop(j)

        x_train = pd.concat(train_group).iloc[:,0:45]
        y_train = pd.concat(train_group).iloc[:,-1] 
        
        log_regr = LogisticRegression(C=c_value, solver = 'liblinear', penalty = 'l1').fit(x_train, y_train)
        
        pred_proba = log_regr.predict_proba(x_test)
        log_loss_ = log_loss(y_test,pred_proba)
        
        cv_score[c_iter][j] = log_loss_
        j = j + 1

    cv_avg[c_iter] = s.mean(cv_score[c_iter]) # mean of C-value
    c_iter = c_iter + 1
    
# plot log-loss over C
meanpointprops = dict(marker='D', markeredgecolor='black',markerfacecolor='firebrick', markersize=2) 
df = pd.DataFrame(cv_score.transpose(), columns=param_grid['C'])
ax = sns.boxplot(x="variable", y="value", data=pd.melt(df), showmeans=True, meanprops=meanpointprops)
ax.axes.set_title("Log-Loss as C changes", fontsize=16)
ax.set_xlabel("C values", fontsize=10)
ax.set_ylabel("Log-loss", fontsize=10)
ax.xaxis.set_major_locator(ticker.MaxNLocator(20))
plt.setp(ax.get_xticklabels(), rotation=45)
plt.show()

# To find the best C-Value
minElement = np.amin(cv_avg) 
result = np.where(cv_avg == np.amin(cv_avg))
result_index = result[0][0]

# Refitting model using chosen C value 
X_train_set = Q1_data.iloc[:500,0:45]
Y_train_set = Q1_data.iloc[:500,-1]

X_test_set = Q1_data.iloc[500:,0:45]
Y_test_set = Q1_data.iloc[500:,-1]

log_regr_refit = LogisticRegression(C=param_grid['C'][result_index], solver = 'liblinear', penalty = 'l1').fit(X_train_set, Y_train_set)
y_pred = log_regr_refit.predict(X_test_set)
y_pred_train = log_regr_refit.predict(X_train_set)

print("Test Accuracy:",accuracy_score(Y_test_set, y_pred))
print("Train Accuracy:",accuracy_score(Y_train_set, y_pred_train))
print("Best C Value is:", param_grid['C'][result_index])
