import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from pandasai import SmartDataframe
from langchain_groq import ChatGroq
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from datetime import timedelta
import json
import base64 
from pandasai import Agent
import sklearn
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Loading environment variables from .env file
load_dotenv() 

# Function to chat with CSV data
def chat_with_csv(df,query):
    # Loading environment variables from .env file
    load_dotenv() 
    
    # Function to initialize conversation chain with GROQ language model
    groq_api_key = os.environ['GROQ_API_KEY']

    # Initializing GROQ chat with provided API key, model name, and settings
    llm = ChatGroq(
    groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile",
    temperature=0.0)
    # Initialize SmartDataframe with DataFrame and LLM configuration
    pandas_ai = SmartDataframe(df, config={"llm": llm})
    # Chat with the DataFrame using the provided query
    result = pandas_ai.chat(query)
    return result

# Set layout configuration for the Streamlit page
st.set_page_config(layout='wide')
# Set title for the Streamlit application
st.title("Data Analysis ChatBot")


# Upload multiple CSV files
input_csvs = st.sidebar.file_uploader("Upload your CSV files", type=['csv'], accept_multiple_files=True)

  
# Check if CSV files are uploaded
if input_csvs:
    # Select a CSV file from the uploaded files using a dropdown menu
    selected_file = st.selectbox("Select a CSV file", [file.name for file in input_csvs])
    selected_index = [file.name for file in input_csvs].index(selected_file)

    st.info("CSV uploaded successfully")
data = pd.read_csv(input_csvs[selected_index])
st.dataframe(data.head(3), use_container_width=True)

# Convert the 'date' column to datetime, using NaT for parsing errors.
data['date'] = pd.to_datetime(data['date'], errors='coerce')

# Optionally fill missing dates with np.nan or a default date if needed
# For example, you could set missing dates to a specific default date:
# data['date'] = data['date'].fillna(pd.Timestamp("2000-01-01"))

# Add 30 days to the 'date' column.
data['date_plus_offset'] = data['date'] + timedelta(days=30)

# Replace any instance of pd.NA in the entire DataFrame with np.nan
data = data.replace({pd.NA: np.nan})

# Display the updated dataframe
st.dataframe(data.head(3), use_container_width=True))

    agent = Agent(data, config={
    "custom_whitelisted_dependencies": ["scikit-learn","statsmodels", "scipy", "ployfit","prophet","sklearn"]
    })

    #Enter the query for analysis
    st.info("Chat Below")
    input_text = st.text_area("Enter the query")

    #Perform analysis
    if input_text:
        if st.button("Chat with Bot"):
            st.info("My Query: "+ input_text)
            result = chat_with_csv(data,input_text)
            st.success(result)
  if st.sidebar.button("Charts"):
    # List of tuples containing (image path, caption)
    chart_images = [
        ("/mount/src/dataanalysisbot/exports/charts/temp_chart.png", "Temporary Chart"),
        ("/mount/src/dataanalysisbot/exports/charts/chart2.png", "Chart 2"),
        ("/mount/src/dataanalysisbot/exports/charts/chart3.png", "Chart 3"),
        ("/mount/src/dataanalysisbot/exports/charts/chart4.png", "Chart 4"),
        ("/mount/src/dataanalysisbot/exports/charts/chart5.png", "Chart 5"),
        # Additional charts can be added here if needed
    ]
    
    # Limit to a maximum of 5 charts
    for image_path, caption in chart_images[:5]:
        st.image(image_path, caption=caption)

def hide_streamlit_and_github_logos():
    hide_css = """
    <style>
        /* Hide the hamburger menu */
        #MainMenu {visibility: hidden;}
        /* Hide the header */
        header {visibility: hidden;}
        /* Hide the footer (including "Made with Streamlit" and any GitHub links) */
        footer {visibility: hidden;}
        /* Alternatively, if you want to remove only the links in the footer, use:
        footer a {display: none !important;}
        */
    </style>
    """
    st.markdown(hide_css, unsafe_allow_html=True)
    hide_streamlit_and_github_logos()




     
