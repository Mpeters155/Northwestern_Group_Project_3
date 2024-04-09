# Import libraries
import pandas as pd
import numpy as np
from nixtlats import TimeGPT
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import hvplot.pandas
import yfinance as yf
import datetime
import streamlit as st # type: ignore
from dataclasses import dataclass
from typing import Any, List
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
 

if "ticker_df" not in st.session_state:
    st.session_state["ticker_df"] = pd.DataFrame(columns=["Date", "Close Price"])

if "pred_df" not in st.session_state:
    st.session_state["pred_df"] = pd.DataFrame(columns=["Date", "Close Price"])

if "forecast" not in st.session_state:
    st.session_state["forecast"] = pd.DataFrame(columns=["Date", "Close Price"])


# Function to fetch data from yfinance
drop_cols = ['Open', 'High', 'Low','Volume', 'Dividends', 'Stock Splits', 'Capital Gains']
def get_history(fund_ticker):
    tckr_name = yf.Ticker(fund_ticker)
    ticker_df = tckr_name.history(period = '10y')
    ticker_df = ticker_df.drop(columns= drop_cols)
    ticker_df = ticker_df.dropna()
    ticker_df.index = ticker_df.index.date
    ticker_df.index.rename('Date', inplace = True)
    ticker_df.reset_index(inplace=True)
    ticker_df.rename(columns = {"Close": "Close Price"}, inplace = True)
    return ticker_df

# Function to run TimeGpt model
def run_gpt(ticker_df):
    # Loading .env environment variables
    load_dotenv()
    # Setting variable for TimeGPT_Token
    TimeGPT_Token = os.getenv("TIMEGPT_TOKEN")
    # Instantiating TimeGPT Model
    timegpt = TimeGPT(token = TimeGPT_Token)
    # Validating TimeGPT token
    timegpt.validate_token()
    fcst_df = timegpt.forecast(df = ticker_df, 
                               time_col = 'Date', 
                               target_col = 'Close Price',  
                               h=252*2,
                               freq = "B",
                               fewshot_steps = 10, 
                               fewshot_loss = 'mse',
                               model='long-horizon'
                               )
    pred_df = pd.concat([ticker_df,fcst_df], axis = 0, join = 'outer')
    pred_df['Close Price'].fillna(pred_df['TimeGPT'], inplace= True)
    pred_df.reset_index(inplace = True)
    pred_df.drop(columns=["index", 'TimeGPT'], inplace = True)
    return pred_df

# Defining function to run the Prophet Model
#df = pd.read_csv('./VTSAX.csv')
def run_prophet(ticker_df):
    prophet_df = ticker_df[['Date','Close Price']]
    # Rename the features: These names are required for the model fitting
    prophet_df = prophet_df.rename(columns = {'Date':'ds','Close Price':'y'})
    #Create Prophet model
    model = Prophet(daily_seasonality = True)
    #Fit the model
    model.fit(prophet_df)
    #Specify the number of days in future
    fut = model.make_future_dataframe(periods=252*2)
    # predict the model
    forecast = model.predict(fut)
    #Clean Data for Visualization
    forecast = forecast[['ds', 'yhat']]
    forecast.rename(columns = {'ds':'Date', 'yhat':'Close Price'}, inplace=True)
    #Return the forecast DataFrame
    return forecast
# Plotting the forecast
#plot_plotly(forecast)


# Database for Mutual Fund information
mutual_fund_data = {
    'Vanguard Total Stock Market Index Fund Admiral Shares': [
        'Vanguard Total Stock Market Index Fund Admiral Shares',
        'VTSAX',
        '$1.3 trillion (Feb. 28, 2021)',
        '$3,000',
        'Apple, Microsoft, Amazon, Alphabet, and Tesla',
        '11/13/2000'],

    'Vanguard 500 Index Fund Admiral Shares': [
        'Vanguard 500 Index Fund Admiral Shares',
        'VFIAX',
        '$808.8 billion (Feb. 28, 2022)',
        '$3,000',
        'Apple, Microsoft, Amazon, Alphabet, Tesla, Nvidia, and Berkshire Hathaway Inc',
        'Aug 31, 1976'],

    'Vanguard Total International Stock Index Fund Admiral Shares':[
        'Vanguard Total International Stock Index Fund Admiral Shares',
        'VTIAX',
        '$385.5 billion (as of Feb. 28, 2022)',
        '$3,000',
        'Taiwan Semiconductor Manufacturing, Nestle SA, Samsung Electronics, and Toyota Motor Corporation',
        'Jan 26, 2011'],

    'Fidelity 500 Index Fund':[
        'Fidelity 500 Index Fund',
        'FXAIX',
        '$399 billion (as of Mar. 31, 2022)',
        '$0',
        'Apple, Microsoft, Amazon, Meta, and Alphabet',
        'May 04, 2011'],

    'Vanguard Total Bond Market Index Fund Admiral Shares':[
        'Vanguard Total Bond Market Index Fund Admiral Shares',
        'VBTLX',
        '$305.1 billion (as of Feb. 28, 2022)',
        '$3,000',
        "U.S. government bonds with 66.5% of the fund's weighting, while 3.7% are AAA-rated bonds and 3.1% are AA-rated",
        'Dec 11, 1986']
}
# List of Mutual Funds for selection
funds = ['Vanguard Total Stock Market Index Fund Admiral Shares',
         'Vanguard 500 Index Fund Admiral Shares',
         'Vanguard Total International Stock Index Fund Admiral Shares',
         'Fidelity 500 Index Fund',
         'Vanguard Total Bond Market Index Fund Admiral Shares']

### Streamlit Code ###

# Streamlit App Heading
st.markdown("# Mutual Fund Performance Predictor")
st.markdown("## Choose A Mutual Fund to Predict it's future performance.")
st.text(" \n")

# Streamlit sidebar Heading
st.sidebar.markdown("# Mutual Fund Information")
st.sidebar.markdown("---------------------")

# Streamlit Selectbox to choose a Mutual Fund
mutual_fund = st.selectbox("Select a Mutual  Fund", funds)

# Setting Variable to display chosen Fund information
fund_name = mutual_fund_data[mutual_fund][0]
fund_ticker = mutual_fund_data[mutual_fund][1]
fund_asset = mutual_fund_data[mutual_fund][2]
fund_investment = mutual_fund_data[mutual_fund][3]
fund_holdings = mutual_fund_data[mutual_fund][4]
fund_date = mutual_fund_data[mutual_fund][5]

#streamlit code to display information in the sidebar
st.sidebar.write("Fund Name : ", fund_name)
st.sidebar.write("Fund Ticker : ", fund_ticker)
st.sidebar.write("Assets under management : ", fund_asset)
st.sidebar.write("Minimum investment : ", fund_investment)
st.sidebar.write("Holdings include : ", fund_holdings)
st.sidebar.write("Inception Date : ", fund_date)

# Streamlit Button to fetch fund historical data
if st.button("Fetch Historical Data"):
    ticker_df = get_history(fund_ticker)
    st.sidebar.write(ticker_df)
    st.session_state["ticker_df"] = ticker_df

# Button to display plot of historical data
if st.button("Display Historical Data"):
    st.write("Historical Prices for :", fund_ticker)
    st.line_chart(st.session_state["ticker_df"], x='Date', y="Close Price")

# Creating a button to run a machine learning prediction for selected mutual fund 
if st.button("Predict Mutual Fund Performance(using TimeGPT)"):
    with st.spinner("Running TimeGPT Model"):
        pred_df = run_gpt(st.session_state["ticker_df"])
        st.sidebar.write(pred_df)
        st.session_state["pred_df"] = pred_df
    st.success("Model Successfully Ran")

# Button to display plot of the TimeGPT data
if st.button("Display TimeGPT Data"):
    st.line_chart(st.session_state["pred_df"], x='Date', y="Close Price")

#Button to run the Prophet model for selected Mutual Fund
if st.button("Predict Mutual Fund Performance(using Prophet)"):
    with st.spinner("Running Prophet Model"):
        forecast = run_prophet(st.session_state["ticker_df"])
        st.sidebar.write(forecast)
        st.session_state["forecast"] = forecast
    st.success("Model Successfully Ran")

# Button to display plot of the Prophet data
if st.button("Display Prophet Data"):
    st.line_chart(st.session_state["forecast"], x='Date', y="Close Price")

# Setting Tabs to display line charts and dataframes
tab1, tab2 = st.tabs(["Data", "Charts"])
with tab1:
    st.markdown("# Data for Mutual Fund")
    st.write("Historical Data for : ", fund_name)
    st.write(st.session_state["ticker_df"])
    st.write("TimeGPT Data for : ", fund_name)
    st.write(st.session_state["pred_df"])
    st.write("Prophet Data for : ", fund_name)
    st.write(st.session_state["forecast"])
with tab2:
    st.markdown("# Charts for Mutual Fund")
    st.write("Historical Chart for : ", fund_name)
    st.line_chart(st.session_state["ticker_df"], x='Date', y="Close Price")
    st.write("TimeGPT Chart for : ", fund_name)
    st.line_chart(st.session_state["pred_df"], x='Date', y="Close Price")
    st.write("Prophet Chart for : ", fund_name)
    st.line_chart(st.session_state["forecast"], x='Date', y="Close Price")
