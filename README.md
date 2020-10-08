# Regularization

Today you'll be creating several different linear regression models in a predictive machine learning context.

In the cells below, we are importing relevant modules that you might need later on. We also load and prepare the dataset for you.


```python
# Run this cell without changes
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
```


```python
# Run this cell without changes
data = pd.read_csv('raw_data/advertising.csv', index_col=0)
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
# Run this cell without changes
X = data.drop('sales', axis=1)
y = data['sales']
```


```python
# Run this cell without changes
X_train , X_test, y_train, y_test = train_test_split(X, y,random_state=2019)
```

To make things simpler for the following examples, we will train models on `X_train` and `y_train`, and evaluate them on `X_test` and `y_test`, without the creation of any additional holdout validation set.

### 1. We'd like to create linear regression models with a bit of added complexity, and we will do it by adding some polynomial terms. Write a function to calculate train and test error for different polynomial degrees.

This function should:
* take `degree` as a parameter that will be used to create polynomial features to be used in a linear regression model
* create a PolynomialFeatures object for that degree and fit a linear regression model using the transformed data
* calculate the mean square error for each level of polynomial
* return the `train_error` and `test_error`

Hint: use [PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html) from SciKit-Learn to preprocess the `X` values.



```python
def polynomial_regression(degree, X_train , X_test, y_train, y_test):
    """
    Calculate train and test error for a linear regression with polynomial features.
    
    input: Polynomial degree, X for training, X for testing, y for training,
    y for testing
    output: A two-tuple containing the mean squared error for train and test set
    """
    # 1. Instantiate a PolynomialFeatures object and create transformed
    # versions of X_train and X_test
    
    # 2. Instantiate a linear regression and fit it to the transformed data
    
    # 3. Calculate the train and test MSEs, and return them as a tuple
    
    None
### BEGIN SOLUTION
    poly = PolynomialFeatures(degree=degree,interaction_only=False)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    lr_poly = LinearRegression()
    lr_poly.fit(X_poly_train,y_train)
    train_error = mean_squared_error(y_train, lr_poly.predict(X_poly_train))
    test_error = mean_squared_error(y_test, lr_poly.predict(X_poly_test))
    return train_error, test_error

from test_scripts.test_class import Test
test = Test()

degree_3 = polynomial_regression(3, X_train, X_test, y_train, y_test)
degree_4 = polynomial_regression(4, X_train, X_test, y_train, y_test)

test.save(degree_3, "degree_3")
test.save(degree_4, "degree_4")
### END SOLUTION

# Use this code to test your solution
print("""
+ ------- + ----------- + ---------- +
| Degrees | Train Error | Test Error |
+ ------- + ----------- + ---------- +""")
for degree in range(1, 9):
    train_error, test_error = polynomial_regression(degree, X_train, X_test, y_train, y_test)
    print("| {:^7} | {:11.4f} | {:10.4f} |".format(degree, train_error, test_error))
print("+ ------- + ----------- + ---------- +")
```

    
    + ------- + ----------- + ---------- +
    | Degrees | Train Error | Test Error |
    + ------- + ----------- + ---------- +
    |    1    |      2.6669 |     3.3856 |
    |    2    |      0.4283 |     0.1864 |
    |    3    |      0.2424 |     0.1528 |
    |    4    |      0.1818 |     1.9523 |
    |    5    |      0.0651 |     5.6725 |
    |    6    |      0.0459 |    22.4071 |
    |    7    |      0.0313 |    24.1634 |
    |    8    |      0.0391 |   718.2957 |
    + ------- + ----------- + ---------- +



```python
# PUT ALL WORK FOR THE ABOVE QUESTION ABOVE THIS CELL
# THIS UNALTERABLE CELL CONTAINS HIDDEN TESTS

degree_3 = polynomial_regression(3, X_train, X_test, y_train, y_test)

# The result of this function call should be a tuple containing two
# floating point numbers
assert type(degree_3) == tuple
assert len(degree_3) == 2
assert type(degree_3[0]) == np.float64 or type(degree_3[0]) == float

### BEGIN HIDDEN TESTS

from test_scripts.test_class import Test
test = Test()

degree_4 = polynomial_regression(4, X_train, X_test, y_train, y_test)

test.run_test(degree_3, "degree_3")
test.run_test(degree_4, "degree_4")

### END HIDDEN TESTS
```

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

### 2. Refer to the plot of train and test errors for an example model above. What is the optimal number of degrees for our polynomial features in this model?

### In general, how does increasing the polynomial degree relate to the Bias/Variance tradeoff?  (Note that this graph shows RMSE and not MSE.)

=== BEGIN MARK SCHEME ===

The optimal number of features in this example is 3 because the testing error is minimized at this point, and it increases dramatically with a higher degree polynomial.

As we increase the polynomial features, it is going to cause our training error to decrease, which decreases the bias but increases the variance (the testing error increases).

In other words, the more complex the model, the higher the chance of overfitting.

=== END MARK SCHEME ===

### 3. In general what methods would you can use to reduce overfitting and underfitting? Provide an example for both and explain how each technique works to reduce the problems of underfitting and overfitting.

=== BEGIN MARK SCHEME ===

Overfitting: Regularization. With regularization, more complex models are penalized. This ensures that the models are not trained to too much "noise."

Underfitting: Feature engineering. By adding additional features, you enable your machine learning models to gain insights about your data.

=== END MARK SCHEME ===

### 4. What are the two types of regularization for linear regression, and what is the difference between them?

=== BEGIN MARK SCHEME ===

L1 or Lasso Regression adds a term to the cost function which reduces some smaller weights down to zero.

L2 or Ridge Regression adds a term to the cost function which penalizes weights based on their size, bringing all of them closer to zero.

=== END MARK SCHEME ===

### 5. Why is scaling input variables a necessary step before regularization?

=== BEGIN MARK SCHEME ===

Regularization adjusts feature weights depending on their magnitude.

Feature weights themselves depend on both the feature importance and the magnitude of the input variable.

Therefore, it's important to control for the magnitude of the input variable by scaling all features the same.

=== END MARK SCHEME ===
