# Regularized Logistic Regression and the Bootstrap
Consider the dataset with binary response variable Y , and 45 continuous features X1, . . . , X45. Regularized Logistic Regression is a regression model used when the response variable is binary valued. Instead of using mean squared error loss as in standard regression problems, we instead minimize the log-loss (aka cross entropy loss). 

For a parameter vector β = (β1,...,βp) ∈ R^p, y_i ∈ {0,1}, x_i ∈ Rp for i = 1,...,n, the log-loss is

![Screen Shot 2021-08-27 at 2 39 41 pm](https://user-images.githubusercontent.com/43845085/131072660-72695f3a-5edf-44c1-9356-c5f0ecfad9f6.png)

where s(z) is the logistic sigmoid.

![Screen Shot 2021-08-27 at 2 41 30 pm](https://user-images.githubusercontent.com/43845085/131072795-c0290aeb-361e-4885-bf88-6c6300deaf48.png)

In practice, we will usually add a penalty term, and consider the optimization: 

![image](https://user-images.githubusercontent.com/43845085/131073295-45d8adc3-8c15-46b3-b66c-bdfd1ee11fd8.png)

## Part 1 - Cross Validation Implementation over the choice of C (from scratch)
The value of C that gave the best performance in terms of average log-loss over 10-fold cross validation was when C = 0.1879. The logistic regression model was then refitted with this C value. The test and train accuracy at C=0.1879 are as follows.
- Test Accuracy: 0.74
- Train Accuracy: 0.75

![Screen Shot 2021-08-27 at 3 02 32 pm](https://user-images.githubusercontent.com/43845085/131074455-9de578d3-2b2b-4b3e-be7e-2dc65aa5a504.png)

## Part 2 - NonParametric 90% Bootstrap Confidence Interval Implementation (from scratch)
A 90% confidence interval gives us a range of values for which we believe with 90% probability that the true parameter is contained in that interval. If the computed 90% confidence interval contains the value zero, then this suggests to us that the feature should not be included in our model. 

![Screen Shot 2021-08-27 at 3 07 02 pm](https://user-images.githubusercontent.com/43845085/131074812-dfa6c7a8-6427-44e0-98b5-93fad13512db.png)

The intervals constructed shown above tell us that most of the coefficients in the model are plausibly zero (indicated by the red lines). It is therefore a good idea to regularize the logistic regression fit. We would likely need a value of C << 1 to get a model that is sparse and doesn’t include many of the noisy features.
