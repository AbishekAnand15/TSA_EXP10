### DEVELOPED BY: Abishek Xavier A
### REGISTER NO: 212222230004
### DATE:
# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load the furniture sales data
data = pd.read_csv('/content/Super_Store_data.csv', encoding='ISO-8859-1')
data['Order Date'] = pd.to_datetime(data['Order Date'])

# Aggregate sales by date
data['date'] = data['Order Date'].dt.date
daily_sales = data.groupby('date')['Sales'].sum().reset_index()

# Set date as the index
daily_sales['date'] = pd.to_datetime(daily_sales['date'])
daily_sales.set_index('date', inplace=True)

# Plot the daily furniture sales
plt.plot(daily_sales.index, daily_sales['Sales'])
plt.xlabel('Date')
plt.ylabel('Furniture Sales ($)')
plt.title('Furniture Sales Time Series')
plt.show()

# Check stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

check_stationarity(daily_sales['Sales'])

# Plot ACF and PACF
plot_acf(daily_sales['Sales'])
plt.show()
plot_pacf(daily_sales['Sales'])
plt.show()

# Split the data into training and testing sets
train_size = int(len(daily_sales) * 0.8)
train, test = daily_sales['Sales'][:train_size], daily_sales['Sales'][train_size:]

# Fit the SARIMA model
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

# Generate predictions
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate RMSE
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Plot actual vs predicted sales
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Furniture Sales ($)')
plt.title('SARIMA Model Predictions for Furniture Sales')
plt.legend()
plt.xticks(rotation=45)
plt.show()


```
### OUTPUT:

![Screenshot 2024-11-11 104257](https://github.com/user-attachments/assets/e2823411-93e2-495f-9d1d-c862c04c4887)
![Screenshot 2024-11-11 104305](https://github.com/user-attachments/assets/1b801344-7c26-4925-a84a-9123e90b2dd1)
![Screenshot 2024-11-11 104315](https://github.com/user-attachments/assets/098ac74e-fdc7-4f03-9e78-edd40b16552f)
![Screenshot 2024-11-11 104323](https://github.com/user-attachments/assets/3785289f-0fa4-4a10-9355-5a3f4ba9eb62)






### RESULT:
Thus the program run successfully based on the SARIMA model.
