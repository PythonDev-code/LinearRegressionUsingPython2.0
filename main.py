import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
file_path = 'C:/Users/dhanr/PycharmProjects/LinearRegressionUsingPython2.0/ACM_Clean.csv'
def load_stock_data(file_path):
    data = pd.read_csv(file_path)
    datetime.datetime.strptime("Date", '%m-%d-%Y').strftime('%m-%d-%y')

    data["date"] = pd.to_datetime(data["Date"])
    data = data.set_index("date")
    data = data.fillna(method="ffill")
    return data
def fit_linear_regression(data):

    X = data.index.values.reshape(-1, 1)
    y = data["Adj Close"].values
    model = LinearRegression()
    model.fit(X, y)
    return model
def fit_polynomial_regression(data, degree):

    X = data.index.values.reshape(-1, 1)
    y = data["Adj Close"].values
    model = PolynomialRegression(degree=degree)
    model.fit(X, y)
    return model
def fit_arima_model(data):
    model = ARIMA(data["price"], order=(1, 1, 1))
    model_fit = model.fit()
    return model_fit
def plot_stock_data(data, linear_model, poly_model, arima_model):

    plt.figure(figsize=(12, 8))

    # Plot the actual stock prices
    plt.scatter(data.index, data["price"], label="Actual Prices")
    # Plot the linear regression model
    plt.plot(data.index, linear_model.predict(data.index.values.reshape(-1,
    1)), color="r", label="Linear Regression")
    # Plot the polynomial regression model
    plt.plot(data.index, poly_model.predict(data.index.values.reshape(-1,
    1)), color="g", label="Polynomial Regression")
    # Plot the ARIMA model
    plt.plot(data.index, arima_model.fittedvalues, color="b", label="ARIMA")
    plt.title("Stock Price Data and Models")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.show()
# Example usage
data = load_stock_data(file_path)
linear_model = fit_linear_regression(data)
poly_model = fit_polynomial_regression(data, degree=2)
arima_model = fit_arima_model(data)
plot_stock_data(data, linear_model, poly_model, arima_model)