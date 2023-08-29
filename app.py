# app.py

import streamlit as st
import yfinance as yf
import numpy as np

# Title of the app
st.title("Stock Price App")

# Create a text input for the stock ticker
ticker = st.text_input("Enter a stock ticker:", "AAPL").upper()

# Fetch data
def load_data(ticker):
    data = yf.Ticker(ticker)
    history = data.history(period="1y")
    return history

if ticker:
    data_load_state = st.text('Loading data...')
    data = load_data(ticker)
    data_load_state.text('Loading data... Done!')
    
    # Calculate KPIs
    mean_close = np.mean(data['Close'])
    max_close = np.max(data['Close'])
    min_close = np.min(data['Close'])
    mean_volume = np.mean(data['Volume'])
    data['Daily Return'] = data['Close'].pct_change()
    mean_daily_return = np.mean(data['Daily Return'])
    volatility = np.std(data['Daily Return'])

    # Display KPIs in a table
    kpis = {
        'Mean Closing Price': mean_close,
        'Max Closing Price': max_close,
        'Min Closing Price': min_close,
        'Mean Volume': mean_volume,
        'Mean Daily Return': mean_daily_return,
        'Volatility': volatility
    }
    st.subheader(f"KPIs for {ticker} over the last year")
    st.table(kpis.items())

    # Layout for two plots side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Closing Price for {ticker} over the last year")
        st.line_chart(data['Close'])
        
    with col2:
        st.subheader(f"30-Day Moving Average for {ticker}")
        data['MA30'] = data['Close'].rolling(window=30).mean()
        st.line_chart(data[['Close', 'MA30']])
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader(f"Daily Returns for {ticker}")
        st.line_chart(data['Daily Return'], use_container_width=True)

    with col4:
        st.subheader(f"Volume for {ticker} over the last year")
        st.line_chart(data['Volume'])

