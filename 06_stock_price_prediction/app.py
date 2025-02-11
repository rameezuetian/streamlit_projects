import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.title('Stock Market Forecasting App')
st.subheader('This app forecasts stock market prices based on selected parameters.')
st.image("https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg")

st.sidebar.header('Select Parameters')
start_date = st.sidebar.date_input('Start date', date(2020, 1, 1))
end_date = st.sidebar.date_input('End date', date(2020, 12, 31))
ticker_list = ["AAPL", "MSFT", "GOOG", "META", "TSLA", "NVDA", "ADBE", "PYPL", "INTC"]
ticker = st.sidebar.selectbox('Select Company', ticker_list)

if start_date >= end_date:
    st.error("Error: End date must be after start date.")
else:
    data = yf.download(ticker, start=start_date, end=end_date)
    data.insert(0, "Date", data.index, True)
    data.reset_index(drop=True, inplace=True)
    st.write(f'Data from {start_date} to {end_date}')
    st.write(data)

    st.header('Data Visualization')
    st.subheader('Stock Price Over Time')
    fig = px.line(data, x='Date', y='Close', title='Closing Price of the Stock', width=1000, height=600)
    st.plotly_chart(fig)

    column = st.selectbox('Select Column for Forecasting', data.columns[1:])
    data = data[['Date', column]]
    st.write("Selected Data")
    st.write(data)

    st.header('Stationarity Check')
    p_value = adfuller(data[column])[1]
    st.write(f"P-Value: {p_value}")
    if p_value < 0.05:
        st.success("The data is stationary.")
    else:
        st.warning("The data is not stationary.")

    st.header('Data Decomposition')
    decomposition = seasonal_decompose(data[column], model='additive', period=12)
    st.plotly_chart(px.line(x=data["Date"], y=decomposition.trend, title='Trend'))
    st.plotly_chart(px.line(x=data["Date"], y=decomposition.seasonal, title='Seasonality'))
    st.plotly_chart(px.line(x=data["Date"], y=decomposition.resid, title='Residuals'))

    models = ['SARIMA', 'Random Forest', 'LSTM', 'Prophet', 'XGBoost', 'Moving Average', 'ARIMA']
    selected_model = st.sidebar.selectbox('Select Model for Forecasting', models)

    if selected_model == 'SARIMA':
        p = st.slider('p', 0, 5, 2)
        d = st.slider('d', 0, 5, 1)
        q = st.slider('q', 0, 5, 2)
        seasonal_order = st.number_input('Seasonal Order', 0, 24, 12)
        forecast_period = st.number_input('Forecast Days', 1, 365, 10)
        model = sm.tsa.statespace.SARIMAX(data[column], order=(p, d, q), seasonal_order=(p, d, q, seasonal_order)).fit()
        predictions = model.get_prediction(start=len(data), end=len(data) + forecast_period).predicted_mean
        predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
        st.write("Predictions", predictions)
        fig = go.Figure([go.Scatter(x=data["Date"], y=data[column], name='Actual'),
                          go.Scatter(x=predictions.index, y=predictions, name='Predicted')])
        st.plotly_chart(fig)

    elif selected_model == 'Random Forest':
        train_size = int(len(data) * 0.8)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
        rf_model.fit(np.arange(train_size).reshape(-1, 1), data[column][:train_size])
        predictions = rf_model.predict(np.arange(train_size, len(data)).reshape(-1, 1))
        st.write(f"RMSE: {np.sqrt(mean_squared_error(data[column][train_size:], predictions))}")
        fig = go.Figure([go.Scatter(x=data['Date'], y=data[column], name='Actual'),
                          go.Scatter(x=data['Date'][train_size:], y=predictions, name='Predicted')])
        st.plotly_chart(fig)

    elif selected_model == 'ARIMA':
        p = st.slider('AR Order (p)', 0, 5, 2)
        d = st.slider('Differencing (d)', 0, 5, 1)
        q = st.slider('MA Order (q)', 0, 5, 2)
        forecast_period = st.number_input('Forecast Days', 1, 365, 10)
        model = sm.tsa.ARIMA(data[column], order=(p, d, q)).fit()
        predictions = model.forecast(steps=forecast_period)
        future_dates = pd.date_range(start=end_date, periods=forecast_period, freq='D')
        fig = go.Figure([go.Scatter(x=data['Date'], y=data[column], name='Actual'),
                          go.Scatter(x=future_dates, y=predictions, name='Predicted')])
        st.plotly_chart(fig)

    elif selected_model == 'Prophet':
        prophet_data = data.rename(columns={'Date': 'ds', column: 'y'})
        prophet_model = Prophet()
        prophet_model.fit(prophet_data)
        future = prophet_model.make_future_dataframe(periods=365)
        forecast = prophet_model.predict(future)
        st.plotly_chart(px.line(x=forecast['ds'], y=forecast['yhat'], title='Forecast with Prophet'))

    elif selected_model == 'Moving Average':
        window = st.slider('Select Window Size', 5, 50, 20)
        ma_predictions = data[column].rolling(window=window).mean()
        st.plotly_chart(go.Figure([go.Scatter(x=data['Date'], y=ma_predictions, name='Moving Average')]))
