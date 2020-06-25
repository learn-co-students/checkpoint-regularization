# Regularization

Today you'll be creating several different linear regression models in a predictive machine learning context.

In the cells below, we are importing relevant modules that you might need later on. We also load and prepare the dataset for you.


```python
# Run this cell without changes
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
# Run this cell without changes
data = pd.read_csv('raw_data/advertising.csv').drop('Unnamed: 0',axis=1)
data.describe()
```


```python
# Run this cell without changes
X = data.drop('sales', axis=1)
y = data['sales']
```


```python
# Run this cell without changes
# splits the data into training and testing set. Do not change the random state please!
X_train , X_test, y_train, y_test = train_test_split(X, y,random_state=2019)
```

### 1. We'd like to add a bit of complexity to the model created in the example above, and we will do it by adding some polynomial terms. Write a function to calculate train and test error for different polynomial degrees.

This function should:
* take `degree` as a parameter that will be used to create polynomial features to be used in a linear regression model
* create a PolynomialFeatures object for each degree and fit a linear regression model using the transformed data
* calculate the mean square error for each level of polynomial
* return the `train_error` and `test_error` 



```python
def polynomial_regression(degree):
    """
    Calculate train and test error for a linear regression with polynomial features.
    (Hint: use PolynomialFeatures)
    
    input: Polynomial degree
    output: Mean squared error for train and test set
    """
    # Your code here
    
    # Replace None with appropriate code
    train_error = None
    test_error = None
    return train_error, test_error
```

#### Try out your new function


```python
# Run this cell without changes
polynomial_regression(3)
```


```python
# Run this cell without changes
polynomial_regression(4)
```

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
# Your written answer here
```

### 3. In general what methods would you can use to reduce overfitting and underfitting? Provide an example for both and explain how each technique works to reduce the problems of underfitting and overfitting.


```python
# Your written answer here
```

### 4. What is the difference between the two types of regularization for linear regression?


```python
# Your written answer here
```

### 5. Why is scaling input variables a necessary step before regularization?


```python
# Your written answer here
```
