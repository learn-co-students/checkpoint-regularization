# Regularization

Today you'll be creating several different linear regression models in a predictive machine learning context.

In the cells below, we are importing relevant modules that you might need later on. We also load and prepare the dataset for you.


```python
import pandas as pd
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
```


```python
data = pd.read_csv('raw_data/advertising.csv').drop('Unnamed: 0',axis=1)
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TV</th>
      <th>radio</th>
      <th>newspaper</th>
      <th>sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>200.000000</td>
      <td>200.000000</td>
      <td>200.000000</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>147.042500</td>
      <td>23.264000</td>
      <td>30.554000</td>
      <td>14.022500</td>
    </tr>
    <tr>
      <th>std</th>
      <td>85.854236</td>
      <td>14.846809</td>
      <td>21.778621</td>
      <td>5.217457</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.700000</td>
      <td>0.000000</td>
      <td>0.300000</td>
      <td>1.600000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>74.375000</td>
      <td>9.975000</td>
      <td>12.750000</td>
      <td>10.375000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>149.750000</td>
      <td>22.900000</td>
      <td>25.750000</td>
      <td>12.900000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>218.825000</td>
      <td>36.525000</td>
      <td>45.100000</td>
      <td>17.400000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>296.400000</td>
      <td>49.600000</td>
      <td>114.000000</td>
      <td>27.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = data.drop('sales', axis=1)
y = data['sales']
```


```python
# split the data into training and testing set. Do not change the random state please!
X_train , X_test, y_train, y_test = train_test_split(X, y,random_state=2019)
```

### 1. We'd like to add a bit of complexity to the model created in the example above, and we will do it by adding some polynomial terms. Write a function to calculate train and test error for different polynomial degrees.

This function should:
* take `degree` as a parameter that will be used to create polynomial features to be used in a linear regression model
* create a PolynomialFeatures object for each degree and fit a linear regression model using the transformed data
* calculate the mean square error for each level of polynomial
* return the `train_error` and `test_error` 



```python

# do a polynomial regression
def polynomial_regression(degree):
    """
    Calculate train and test error for a linear regression with polynomial features.
    (Hint: use PolynomialFeatures)
    
    input: Polynomial degree
    output: Mean squared error for train and test set
    """
    poly = PolynomialFeatures(degree=degree,interaction_only=False)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    lr_poly = LinearRegression()
    lr_poly.fit(X_poly_train,y_train)
    train_error = mean_squared_error(y_train, lr_poly.predict(X_poly_train))
    test_error = mean_squared_error(y_test, lr_poly.predict(X_poly_test))
    return train_error, test_error
```

#### Try out your new function


```python
polynomial_regression(3)
```




    (0.24235967358392094, 0.15281375975888198)




```python
polynomial_regression(4)
```




    (0.18179109323300002, 1.9522597156044699)



#### Check your answers

Approximate MSE for degree 3:
- Train: 0.242
- Test: 0.153

Approximate MSE for degree 4:
- Train: 0.182
- Test: 1.95

### 2. What is the optimal number of degrees for our polynomial features in this model? In general, how does increasing the polynomial degree relate to the Bias/Variance tradeoff?  (Note that this graph shows RMSE and not MSE.)

<img src ="visuals/rsme_poly_2.png" width = "600">

<!---
fig, ax = plt.subplots(figsize=(7, 7))
degree = list(range(1, 10 + 1))
ax.plot(degree, error_train[0:len(degree)], "-", label="Train Error")
ax.plot(degree, error_test[0:len(degree)], "-", label="Test Error")
ax.set_yscale("log")
ax.set_xlabel("Polynomial Feature Degree")
ax.set_ylabel("Root Mean Squared Error")
ax.legend()
ax.set_title("Relationship Between Degree and Error")
fig.tight_layout()
fig.savefig("visuals/rsme_poly.png",
            dpi=150,
            bbox_inches="tight")
--->


```python
# The optimal number of features in this example is 3 because the testing error 
# is minimized at this point, and it increases dramatically with a higher degree polynomial. 
# As we increase the polynomial features, it is going to cause our training error to decrease, 
# which decreases the bias but increases the variance (the testing error increases). 
# In other words, the more complex the model, the higher the chance of overfitting.
```

### 3. In general what methods would you can use to reduce overfitting and underfitting? Provide an example for both and explain how each technique works to reduce the problems of underfitting and overfitting.


```python
# Overfitting: Regularization. With regularization, more complex models are penalized. 
# This ensures that the models are not trained to too much "noise."

# Underfitting: Feature engineering. By adding additional features, you enable your 
# machine learning models to gain insights about your data.
```

### 4. What is the difference between the two types of regularization for linear regression?


```python
# L1 or Lasso Regression adds a term to the cost function which reduces some smaller weights down to zero.
# L2 or Ridge Regression adds a term to the cost function which penalizes weights based on their size,
# bringing all of them closer to zero.
```

### 5. Why is scaling input variables a necessary step before regularization?


```python
# Regularization adjusts feature weights depending on their magnitude.
# Feature weights themselves depend on both the feature importance and the magnitude of the input variable.
# Therefore, it's important to control for the magnitude of the input variable by scaling all features the same.
```
