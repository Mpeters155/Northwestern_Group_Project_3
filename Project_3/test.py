import pandas as pd 
import streamlit as st 
import yfinance as yf
from datetime import date

options = ["AAPL", "GOOGL", "META", "AMZN","TSLA","JNJ","XOM"]

def main():
    # Setting Title
    st.title("Stock Price")

    # Setting Header
    st.subheader("Choose a stock and duration to see graph")

    # Creating a dropdown
    symbol = st.selectbox("Select an option", options)

    # Taking Date Inputs
    startdate = str(st.date_input("Select a start date", date.today()))
    endtdate = str(st.date_input("Select an end date", date.today()))

    # Getting stock Data from yahoo finance
    tickerData = yf.Ticker(symbol)
    tickerDf = tickerData.history(period='1d', start=startdate,end=endtdate)

    submit = st.button("Get Graphs")


    if submit:
        # Ploting Data
        st.markdown("""
        ### Closing Price
        """)
        st.line_chart(tickerDf.Close)

        st.markdown("""
        ### Volume
        """)
        st.line_chart(tickerDf.Volume)
    st.markdown("Developed By Azeem Waqar")
    st.write("""
            [Click here](https://github.com/AzeemWaqarRao/Streamlit-Stock-Price-App) to visit Github Repository.
            """)



if __name__ == "__main__":
    main()