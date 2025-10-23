 201509# Import Libraries
!pip install yfinance

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Fetch Historical Stock Data
data = yf.download('AAPL', start='2015-01-01', end='2025-01-01')
print(data.head())

# Step 2: Prepare Data
data = data[['Close']]
data['Previous Close'] = data['Close'].shift(1)
data.dropna(inplace=True)

X = data[['Previous Close']]
y = data['Close']

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 4: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict
predictions = model.predict(X_test)

# Step 6: Evaluate Model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Step 7: Visualize Predictions
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual Price', color='blue')
plt.plot(predictions, label='Predicted Price', color='red')
plt.title('Stock Price Prediction (Linear Regression)')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()
