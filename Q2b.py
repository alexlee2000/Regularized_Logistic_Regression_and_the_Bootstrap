import numpy as np
from numpy import array
from numpy.linalg import norm
import matplotlib.pyplot as plt

A = np.mat([[1,0,1,-1],
            [-1,1,0,2], 
            [0,-1,-2,1]])
A_t = np.transpose(A)
b = np.mat([[1], 
            [2],
            [3]])
b_t = np.transpose(b)
x_k_prev = np.mat([[1],
                   [1],
                   [1],
                   [1]])

x_values = np.zeros(91)
y_values = np.zeros(91) # we know k = 90 is the last iteration

alpha_k = 0.1
k = 0
converge = False
while (converge == False):
    if(norm(A_t.dot(A.dot(x_k_prev) - b)) < 0.001):
        converge = True
    
    if(0 <= k <= 4 or 86 <= k <= 90): #print at first 5 iteration and last 5 iterations
        print("k =", k,", x_k = [", x_k_prev[0,0],",",x_k_prev[1,0],",",x_k_prev[2,0],",",x_k_prev[3,0],"]")
    
    if(k == 0):
        x_k = x_k_prev - alpha_k*(A_t.dot(A.dot(x_k_prev) - b))
    else:
        # compute alpha_k for x_k+1 
        axb_t = np.transpose(A.dot(x_k_prev) - b)
        aa_tax = ((A.dot(A_t)).dot(A)).dot(x_k_prev)
        aa_tb = (A.dot(A_t)).dot(b)
        aa_ta = (A.dot(A_t)).dot(A)
        a_taxb = A_t.dot(A.dot(x_k_prev) - b)

        alpha_k = (axb_t.dot(aa_tax) - axb_t.dot(aa_tb)) / ((axb_t.dot(aa_ta)).dot(a_taxb))

        # compute x_k+1 using alpha_k and x_k
        x_k = x_k_prev - alpha_k[0,0]*(A_t.dot(A.dot(x_k_prev) - b))
    
    x_values[k] = k
    y_values[k] = alpha_k
    x_k_prev = x_k   
    k = k + 1

# plotting alphas against iterations k 
plt.plot(x_values, y_values)
plt.title('alpha at each iteration')
plt.xlabel('k')
plt.ylabel('alphas')
plt.xticks(np.arange(0, 91, 5))
plt.show()

