# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 04/10/2025
### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv('/content/Tomato.csv', parse_dates=['Date'], index_col='Date')

# ✅ Use 'Average' column as target
col = "Average"
data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop NaN rows
data = data.dropna(subset=[col])
print("Cleaned data head:\n", data.head(), "\nTotal rows:", len(data))

# ✅ ADF Test
adf = adfuller(data[col])
print("ADF:", adf[0], "p-value:", adf[1])

# ✅ Train-test split
train, test = data.iloc[:int(0.8*len(data))], data.iloc[int(0.8*len(data)):]

# ✅ Fit AR model
model = AutoReg(train[col], lags=13).fit()
pred = model.predict(start=len(train), end=len(train)+len(test)-1)

# ✅ MSE
print("MSE:", mean_squared_error(test[col], pred))

# ✅ Plots
plot_acf(data[col], lags=40); plt.show()
plot_pacf(data[col], lags=40); plt.show()
plt.plot(test[col], label="Test")
plt.plot(pred, '--', label="Pred")
plt.legend(); plt.show()

```
### OUTPUT:

<img width="709" height="543" alt="image" src="https://github.com/user-attachments/assets/269083d9-2034-463b-af18-2fef90f87e46" />
<img width="701" height="544" alt="image" src="https://github.com/user-attachments/assets/0e873f36-7386-4b3a-b611-a7b2d389cce9" />
<img width="686" height="521" alt="image" src="https://github.com/user-attachments/assets/fa9c88ee-d454-4f55-ae38-bb70c77e6a1d" />

### RESULT:
Thus we have successfully implemented the auto regression function using python.
