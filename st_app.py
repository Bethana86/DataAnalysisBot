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
    groq_api_key=groq_api_key, model_name="deepseek-r1-distill-qwen-32b",
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

    #load and display the selected csv file 
    st.info("CSV uploaded successfully")
    data = pd.read_csv(input_csvs[selected_index])
    st.dataframe(data.head(3),use_container_width=True)

    agent = Agent(data, config={
    "custom_whitelisted_dependencies": ["scikit-learn","statsmodels", "scipy"]
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




     
