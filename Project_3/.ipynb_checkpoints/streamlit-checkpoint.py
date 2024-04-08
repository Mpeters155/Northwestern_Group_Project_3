import pandas as pd 
import streamlit as st 
import yfinance as yf
from datetime import date

options = ["VTSAX", "VFIAX", "VTIAX", "FXAIX", "VBTLX"]

def main():
    # Title Page
    st.title("Fund Price")

    # Header
    st.subheader("Choose Mutual Fund")

    # Dropdown selection
    symbol = st.selectbox("Select Mutual Fund", options)

library = st.sidebar.selectbox(
    "Which Mutual Fund Do You Want?",
    ("VTSAX", "VFIAX", "VTIAX", "FXAIX", "VBTLX"))


if st.sidebar.button.selectbox("Display Selection"):
    st.sidebar.write(library)
    

if __name__ == "__main__":
    main()

