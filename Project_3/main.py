import streamlit as st

st.set_page_config(
    page_title="Northwestern Fintech Project 3",
    layout="wide")


# Add content to your Streamlit app
st.markdown("# Top 5 Mutual Fund Performance and Forecasting")

st.write(">In this project, we demonstrated the performance of the five largest Mutual Funds and offered the capability to forecast future performance for these funds.")

#add a sidebar for page selection
st.sidebar.success("Select a page above.")

# Create a Streamlit container for the subheader
subheader_container = st.container()

# Define the subheader content
subheader_content = """"
<div class>
<h3>Things You Can Do On This App:</h3>
<ul>
  <li>Forecast Mutual Fund Prices</li>
  <li>View the dataset and interact with Mutual Fund</li>
  <li>Get to know more about this project</li>
</ul>
</div>
"""