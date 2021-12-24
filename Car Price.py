# IMPORTING PACKAGES

import pandas as pd                 # Data processing
import matplotlib.pyplot as plt     # Visualization

from sklearn.model_selection import train_test_split    # Data split
from sklearn.linear_model import LinearRegression       # Linear Regression Algorithm
from sklearn.linear_model import Lasso                  # Lasso regression model
from xgboost import XGBRegressor                        # XGBOOST Model

from sklearn import metrics         # Evaluation metrics

# IMPORTING DATA

df = pd.read_csv('car data.csv')
print(df.head(5))
print('----------------------------')
print()
print(df.shape)
print('----------------------------')
print()

# GETTING SOME INFORMATION ABOUT THE DATASET
print(df.info())
print('----------------------------')
print()

# CHECKING THE NUMBER OF MISSING VALUES
print(df.isnull().sum())
print()

# CHECKING THE DISTRIBUTION OF CATEGORICAL DATA

print(df.Fuel_Type.value_counts())
print('----------------------------')
print()
print(df.Seller_Type.value_counts())
print('----------------------------')
print()
print(df.Transmission.value_counts())
print('----------------------------')
print()

# ENCODING THE CATEGORICAL DATA

# 1. Encoding the fuel type

df.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
df.replace({'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace=True)
df.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)

print(df.head())

# Splitting the data and target
X = df.drop(['Car_Name', 'Selling_Price'], axis=1)
y = df['Selling_Price']
print(X)
print()
print(y)
print()

# Splitting the data into training data and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

print(X.shape, X_train.shape, X_test.shape)
print()
print(y.shape, y_train.shape, y_test.shape)
print()

# Modelling

# 1. Linear Regression Model

# Training data

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_reg_yhat = lin_reg.predict(X_train)

# Testing data

lin_reg1 = LinearRegression()
lin_reg1.fit(X_train, y_train)
lin_reg1_yhat = lin_reg1.predict(X_test)

# 2. Lasso Regression Model

# Training data

lass_reg = Lasso()
lass_reg.fit(X_train, y_train)
lass_reg_yhat = lass_reg.predict(X_train)

# Testing data

lass_reg1 = LinearRegression()
lass_reg1.fit(X_train, y_train)
lass_reg1_yhat = lass_reg1.predict(X_test)

# 3. XGBOOST Regressor Model

# Training data

xgb = XGBRegressor()
xgb.fit(X_train, y_train)
xgb_yhat = xgb.predict(X_train)

# Testing data

xgb1 = XGBRegressor()
xgb1.fit(X_train, y_train)
xgb1_yhat = xgb1.predict(X_test)

# Evaluation

# 1. R Squared Error

print(f'R Squared Error of Linear Regression model on training data is {metrics.r2_score(y_train, lin_reg_yhat)}')
print('------------------------------------------------------------------------')
print()
print(f'R Squared Error of Linear Regression model on testing data is {metrics.r2_score(y_test, lin_reg1_yhat)}')
print('------------------------------------------------------------------------')

print(f'R Squared Error of Lasso Regression model on training data is {metrics.r2_score(y_train, lass_reg_yhat)}')
print('------------------------------------------------------------------------')
print()
print(f'R Squared Error of Lasso Regression model on testing data is {metrics.r2_score(y_test, lass_reg1_yhat)}')
print('------------------------------------------------------------------------')

print(f'R Squared Error of XGB Regression model on training data is {metrics.r2_score(y_train, xgb_yhat)}')
print('------------------------------------------------------------------------')
print()
print(f'R Squared Error of XGB Regression model on testing data is {metrics.r2_score(y_test, xgb1_yhat)}')
print('------------------------------------------------------------------------')

# Visualizing the actual prices and the predicted prices

plt.scatter(y_train, lin_reg_yhat)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Predicted Price")
plt.show()

plt.scatter(y_test, lin_reg1_yhat)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Predicted Price")
plt.show()

plt.scatter(y_train, lass_reg_yhat)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Predicted Price")
plt.show()

plt.scatter(y_test, lass_reg1_yhat)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Predicted Price")
plt.show()

plt.scatter(y_train, xgb_yhat)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Predicted Price")
plt.show()

plt.scatter(y_test, xgb1_yhat)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Predicted Price")
plt.show()
