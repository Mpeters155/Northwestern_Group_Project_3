# %%
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

# %%
drop_cols = ['Volume', 'Dividends', 'Stock Splits', 'Capital Gains']

# %%
def get_history(ticker):
    tckr_name = yf.Ticker(ticker)
    tckr_df = tckr_name.history(period = '10y')
    tckr_df = tckr_df.drop(columns= drop_cols)
    tckr_df = tckr_df.dropna()
    tckr_df.index = tckr_df.index.date
    tckr_df.index.rename('Date', inplace = True)
    tckr_df.reset_index(inplace=True)
    tckr_df.rename(columns = {"Close": "Close Price"}, inplace = True)
    return tckr_df

get_history(ticker)

# %%
tckr_df = get_history(ticker)

# %%
#tckr_df

# %%
# Validating existence of .env file
find_dotenv()

# %%
# Loading .env environment variables
load_dotenv()

# %%
# Setting variable for TimeGPT_Token
TimeGPT_Token = os.getenv("TIMEGPT_TOKEN")

# %%
# Instantiating TimeGPT Model
timegpt = TimeGPT(token = TimeGPT_Token)
# Validating TimeGPT token
timegpt.validate_token()

# %%


# %%
fcst_df = timegpt.forecast(df = tckr_df, time_col = 'Date', target_col = 'Close',  h=12, freq = "MS", fewshot_steps = 10, fewshot_loss = 'mse', model='long-horizon')

# %%
fcst_df.head()

# %%
tckr_df[270:280]

# %%
tckr_df

# %%
fcst_df.head()

# %%
timegpt.plot(fcst_df, time_col='Date', target_col='TimeGPT', engine='plotly')

# %%
fig = fcst_df.hvplot(x='Date')
fig

# %%
pred_df = pd.concat([tckr_df,fcst_df], axis = 0, join = 'outer')

# %%
pred_df

# %%
pred_df['Close'].fillna(pred_df['TimeGPT'], inplace= True)
pred_df

# %%
pred_plot = pred_df.plot(x = 'Date')
pred_plot

# %%
timegpt.plot(pred_df, time_col='Date', target_col='Close', engine='plotly')

# %%



