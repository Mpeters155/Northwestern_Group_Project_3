#Load libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly


# Define functions
def load_data(path):
    dataset = pd.read_csv(path)
    return dataset

# Load the dataset
data_path = ('VTSAX.csv')
load_df = load_data(data_path)

#load test data to see model's performnce
test_df=pd.read_csv('VTSAX.csv')

# Define section
data = st.container()

# Set up the data section that users will interact with
with data:
    data.title("On this page, you can preview the dataset and view daily Fund Price")
    st.write("View the Dataset below")

    # Button to preview the dataset
    if st.button("Preview the dataset"):
        data.write(load_df)

    # Button to view the chart

    st.write("Graph showing daily sales can be viewed below")
    if st.button("View Chart"):

        # Set the "date" column as the index
        load_df = load_df.set_index('date')

        # Display the line chart with dates on the x-axis
        st.subheader("A Chart of the Daily Sales Across Favorita Stores")
        st.line_chart(load_df["sales"])


