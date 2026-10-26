# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries such as NumPy, Pandas, Matplotlib, and the SGDRegressor model from Scikit-learn.
2. Load and preprocess the dataset by separating the input features (independent variables) and the target variables (house price and number of occupants). Split the dataset into training and testing sets.
3. Train the model by creating separate SGDRegressor models for each target variable (house price and number of occupants) and fit them using the training data.
4. Predict and evaluate the results using the testing data, compare the predicted values with the actual values, and calculate performance metrics such as Mean Squared Error (MSE) and R² Score.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Sree Varsha D
RegisterNumber:  212225040422

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
data = fetch_california_housing()
X = data.data[:, :3]   
Y = np.column_stack((data.target, data.data[:, 6]))
print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Example X (first row):", X[0])
print("Example Y (first row):", Y[0])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Train shapes:", X_train.shape, Y_train.shape)
print("Test shapes: ", X_test.shape, Y_test.shape)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)

print("Scaled X_train mean (approx):", X_train_scaled.mean(axis=0))
print("Scaled Y_train mean (approx):", Y_train_scaled.mean(axis=0))
sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)  
multi_output_sgd = MultiOutputRegressor(sgd)

multi_output_sgd.fit(X_train_scaled, Y_train_scaled)
Y_pred_scaled = multi_output_sgd.predict(X_test_scaled)   
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)         
Y_test_orig = scaler_Y.inverse_transform(Y_test_scaled)    

print("First 5 predictions (original units):")
print(Y_pred[:5])

mse = mean_squared_error(Y_test_orig, Y_pred)
print("Mean Squared Error (multi-output):", mse)
​
mse_per_output = np.mean((Y_test_orig - Y_pred) ** 2, axis=0)
print("MSE per output:", mse_per_output)
for i in range(5):
    print(f"Example {i+1}")
    print("Inputs (raw):", X_test[i])
    print("True outputs:", Y_test_orig[i])
    print("Predicted   :", Y_pred[i])
    print("-" * 40)

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = fetch_california_housing()
X, y = data.data[:, :3], data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# SGD Regressor
sgd = SGDRegressor(max_iter=1000, tol=1e-3, eta0=0.01, learning_rate='constant', random_state=42)
sgd.fit(X_train, y_train)
sgd_pred = sgd.predict(X_test)

# Compare
print("LinearRegression MSE:", mean_squared_error(y_test, lr_pred))
print("SGDRegressor MSE:", mean_squared_error(y_test, sgd_pred))

*/
```

## Output:

<img width="1157" height="258" alt="image" src="https://github.com/user-attachments/assets/dcbe03db-3f6b-4b7b-8bdc-f9c044ce809b" />

<img width="1116" height="388" alt="image" src="https://github.com/user-attachments/assets/b6cf8033-0d5c-4c93-b9fe-9cd8b65f3108" />

<img width="1170" height="446" alt="image" src="https://github.com/user-attachments/assets/3be91d66-094e-42cd-8e3f-c76c001fb176" />

<img width="1062" height="590" alt="image" src="https://github.com/user-attachments/assets/b32bc94c-32b3-4e8b-900c-d676a01a5514" />

<img width="1058" height="194" alt="image" src="https://github.com/user-attachments/assets/b78484f4-57bd-4195-acf9-d33eedb64bd6" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
