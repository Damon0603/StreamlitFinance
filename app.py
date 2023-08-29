import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go 
# Constants
RISK_FREE_RATE = 0.0  # Assumption for simplicity

# Title of the app
st.title("Stock Price App")

# Create a text input for the stock ticker
ticker = st.text_input("Enter a stock ticker:", "AAPL").upper()

# Fetch data
def load_data(ticker):
    data = yf.Ticker(ticker)
    history = data.history(period="1y")
    return history

def calculate_sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE):
    return (returns.mean() - risk_free_rate) / returns.std()

def monte_carlo_efficient_frontier(tickers):
    data = {}
    for ticker in tickers:
        data[ticker] = yf.Ticker(ticker).history(period="1y")['Close'].pct_change()
    df = pd.DataFrame(data).dropna()
    
    num_portfolios = 10000
    all_weights = np.zeros((num_portfolios, len(df.columns)))
    ret_arr = np.zeros(num_portfolios)
    vol_arr = np.zeros(num_portfolios)
    sharpe_arr = np.zeros(num_portfolios)

    for ind in range(num_portfolios):
        # Create random weights
        weights = np.array(np.random.random(len(df.columns)))
        weights = weights / np.sum(weights)
        
        # Save weights
        all_weights[ind, :] = weights

        # Expected return
        ret_arr[ind] = np.sum((df.mean() * weights * 252))

        # Expected volatility
        vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(df.cov() * 252, weights)))

        # Sharpe Ratio
        sharpe_arr[ind] = ret_arr[ind] / vol_arr[ind]

    return vol_arr, ret_arr, sharpe_arr

if ticker:
    data_load_state = st.text('Loading data...')
    data = load_data(ticker)
    data_load_state.text('Loading data... Done!')
    
    # Calculate KPIs
    daily_returns = data['Close'].pct_change().dropna()
    mean_daily_return = daily_returns.mean()
    volatility = daily_returns.std()
    sharpe_ratio = calculate_sharpe_ratio(daily_returns)

    # Display KPIs in a table
    kpis = {
        'Mean Closing Price': data['Close'].mean(),
        'Max Closing Price': data['Close'].max(),
        'Min Closing Price': data['Close'].min(),
        'Mean Volume': data['Volume'].mean(),
        'Mean Daily Return': mean_daily_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio
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
        st.line_chart(daily_returns, use_container_width=True)

    with col4:
        st.subheader(f"Volume for {ticker} over the last year")
        st.line_chart(data['Volume'])
    
# Monte Carlo Efficient Frontier
st.subheader("Efficient Frontier with Monte Carlo Simulation")
sample_tickers = ['AAPL', 'MSFT', 'GOOGL']  # Sample stocks for efficient frontier
vol_arr, ret_arr, sharpe_arr = monte_carlo_efficient_frontier(sample_tickers)

# Create a scatter plot with Plotly
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=vol_arr,
        y=ret_arr,
        mode='markers',
        marker=dict(
            size=5,
            color=sharpe_arr, 
            colorscale='Viridis', 
            showscale=True,
            colorbar=dict(title='Sharpe Ratio')
        )
    )
)

fig.update_layout(
    title='Efficient Frontier',
    xaxis_title='Volatility',
    yaxis_title='Expected Return',
    width=700,
    height=400
)

st.plotly_chart(fig)