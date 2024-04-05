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
                               h=24,
                               freq = "MS", 
                               fewshot_steps = 10, 
                               fewshot_loss = 'mse',
                               model='long-horizon'
                               )
    pred_df = pd.concat([ticker_df,fcst_df], axis = 0, join = 'outer')
    pred_df['Close Price'].fillna(pred_df['TimeGPT'], inplace= True)
    pred_df.reset_index(inplace = True)
    pred_df.drop(columns=["index", "TimeGPT"], inplace = True)
    return pred_df

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

def get_fund():
    """Display Mutual Fund Information for user selection"""
    db_list = list(mutual_fund_data.values())

    for number in range(len(funds)):
        st.sidebar.write("Fund Name: ", db_list[number][0])
        st.sidebar.write("Fund Ticker: ", db_list[number][1])
        st.sidebar.write("Assets under management: ", db_list[number][2])
        st.sidebar.write("Minimum investment: ", db_list[number][3])
        st.sidebar.write("Holdings include: ", db_list[number][4])
        st.sidebar.write("Inception Date: ", db_list[number][5])
        st.sidebar.write("------------------------------")

### Streamlit Code ###
# Streamlit App Heading
st.markdown("# Mutual Fund Performance Predictor")
st.markdown("## Choose A Mutual Fund to Predict it's future performance.")
st.text(" \n")

# Streamlit sidebar Heading
st.sidebar.markdown("# Mutual Fund Information")
st.sidebar.markdown("---------------------")

# Number inout to set users initial investment
# init_invst = st.number_input("Set your Initial Investment Amount.")

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
fetch_button = st.button("Fetch Historical Data")
if fetch_button:
    ticker_df = get_history(fund_ticker)
    st.sidebar.write(ticker_df)


#ticker_df = get_history(fund_ticker)

# Button to display plot of historical data
if st.button("Display Historical Data"):
    with st.container():
        st.write("Historical Prices for :", fund_ticker)
        st.line_chart(ticker_df, x='Date', y="Close Price")

# Creating a button to run a machine learning prediction for selected mutual fund
run_button = st.button("Predict Mutual Fund Performance")
if run_button:
    with st.spinner("Running Prediction Model"):
        pred_df = run_gpt(ticker_df)
        st.sidebar.write(pred_df)
    st.success("Model Successfully Ran")
#pred_df = run_gpt(ticker_df)

# Button to display plot of forecasted data
if st.button("Display Predicted Data"):
    st.line_chart(pred_df, x='Date', y="Close Price")
    
# Setting Tabs to display line charts and dataframes
tab1, tab2 = st.tabs(["Data", "Charts"])
with tab1:
    st.markdown("# Data for Mutual Fund")
    st.write("Historical Data for : ", fund_name)
    st.write(ticker_df)
    st.write("Predicted Data for : ", fund_name)
    st.write(pred_df)
with tab2:
    st.markdown("# Charts for Mutual Fund")
    st.write("Historical Chart for : ", fund_name)
    st.line_chart(ticker_df, x='Date', y="Close Price")
    st.write("Predicted Chart for : ", fund_name)
    st.line_chart(pred_df, x='Date', y="Close Price")
