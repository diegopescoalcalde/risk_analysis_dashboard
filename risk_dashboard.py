import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import shapiro, norm
import risk_functions



def home():
    st.title('Home')
    st.header('Welcome! Please read me!')
    st.markdown('''
            Thank you for accessing this simple risk analysis dashboard. Below you can find
            some useful information about the different sections of this app, which you can
            access in the sidebar menu.''')
    with st.expander('* Ticker Info'):
        st.markdown('''
                    This section has charts that provide an overall picture of the ticker prices and returns
                    in the past. 
                * Candlestick Chart \n
                    This chart provides information of open, high, low and close prices for the selected ticker.
                * Returns Line Chart \n
                    The returns line chart presents the daily returns calculated from the prices.
                * Returns Histogram  \n
                    The returns histogram helps us to understand the shape of the return distribution for the ticker,
                    and compares it to the shape of a normal distribution.
                * Shapiro-Wilk Normality Test \n
                    This statistical test calculates the confidence to which the returns distribution can be
                    approximated to a normal distribution. Usually a p-value lower than 0.05 is enough to the
                    assumption of normality. This is a very useful information since normal distribution
                    have many properties that can be used in risk assessment.
                    ''')
        
    with st.expander('* VaR Model Analysis'):
        st.markdown('''
                This section has charts that will help you backtest and visualize the estimated 
                VaR using different models and parameters. It is a helpful tool to understand the
                effect of each model and parameter in the overall risk assessment. The available models are:
                    
                * Historical VaR \n
                    This method calculates VaR from the a percentile of the past return 
                    distribution, assuming that the future return distribution will be similar to the historical one.
                    The two parameters to be defined are the confidence level to be applied and the rolling window from 
                    which the past return distribution will be built. 
            
                * EWMA VaR \n
                    This method also calculates VaR from past returns, but it calculates a 
                    Exponentially Weighted Moving Average, where the earlier values have more weight
                    than older values, in an attempt to appropriately model market heteroscedasticity.
                    The parameters to be defined are the confidence level to be applied and the decay factor
                    lambda. Lower lambda values will put more weight into earlier values and higher lambda will
                    distribute weights more evenly.

                * GARCH VaR \n
                    The GARCH model is a more complex approach that takes into account the conditional
                    autoregressive heteroscedasticity of the volatility. In other words, the estimated
                    volatility is a function of the volatility and the residuals of the previous time steps.
                    The parameters to be defined are the confidence level and the number of lag terms (p,q) in the
                    function, but extensive research has proved that the best estimations are obtained in single
                    lag model, i.e. GARCH(1,1).
                
            ''')
                
    with st.expander('Anomaly Detection'):
        st.markdown('''
                Soon...
                    ''')
                
                
def ticker_info():
    st.title('Ticker Info')

    dict_tickers = {
        'S&P500':'^GSPC',
        'NASDAQ':'^IXIC'
    }

    tickers_options = ['S&P500', 'NASDAQ']

    tickers = st.selectbox('Choose Ticker', tickers_options)

    st.markdown('Select area with mouse to zoom charts, click twice to zoom out.')

    yf_data = yf.download(dict_tickers[tickers], period='5y', interval='1d')

    fig = go.Figure(data=[go.Candlestick(x=yf_data.index,
                                         open=yf_data['Open'],
                                         high=yf_data['High'],
                                         low=yf_data['Low'],
                                         close=yf_data['Close'],
                                         increasing_line_color= 'green', 
                                         decreasing_line_color= 'red')])
    fig.update_layout(xaxis_title='Date',
                      yaxis_title='Price',
                      title=tickers, 
                      xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    returns, fig_close_prices, fig_returns_line, fig_returns_hist, normality_test = risk_functions.compute_returns(tickers, pd.DataFrame(yf_data['Close']))
    #st.plotly_chart(fig_close_prices)
    st.plotly_chart(fig_returns_line, use_container_width=True)
    st.plotly_chart(fig_returns_hist, use_container_width=True)
    st.text('Shapiro-Wilk Normality Test Results')
    st.text(f'Test Statistic:{round(normality_test[0], 5)}')
    st.text(f'P-value:{round(normality_test[1], 5)}')

def model_comparison():
    st.title('VaR Model Analysis')

    dict_tickers = {
        'S&P500':'^GSPC',
        'NASDAQ':'^IXIC'
    }

    tickers_options = ['S&P500', 'NASDAQ']

    tickers = st.selectbox('Choose Ticker', tickers_options)

    yf_data = yf.download(dict_tickers[tickers], period='5y', interval='1d')
    returns, fig_close_prices, fig_returns_line, fig_returns_hist, normality_test = risk_functions.compute_returns(tickers, pd.DataFrame(yf_data['Close']))
    models = st.multiselect('Select Risk Models', ['Historical VaR', 'EWMA VaR', 'GARCH VaR'], ['EWMA VaR'])
    var_list = []
    error_list = []
    confidence_level = st.number_input('Confidence Level', min_value=0.00, max_value=1.00, value=0.95)
    if 'EWMA VaR' in models:
        lambda_value = st.number_input('Lambda Value', min_value=0.00, max_value=1.00, value=0.94)
        returns = risk_functions.compute_ewma_var(returns, lambda_value=lambda_value, time_horizon=1, confidence_level=confidence_level)
        var_list.extend(['EWMA VaR', 'EWMA VaR - Negative Returns'])
        error_list.extend(['EWMA VaR - Total Percentage Error', 'EWMA VaR - 365D Rolling Percentage Error'])
    if 'Historical VaR' in models:
       window = st.number_input('Window Value in Days', min_value=1, max_value=2000, value=1095)
       returns = risk_functions.calculate_historical_var(returns, window=window, confidence_level=confidence_level)
       var_list.extend(['Historical VaR', 'Historical VaR - Negative Returns'])
       error_list.extend(['Historical VaR - Total Percentage Error', 'Historical VaR - 365D Rolling Percentage Error'])
    if 'GARCH VaR' in models:
        model_results, garch_var, returns = risk_functions.compute_garch_var(returns, p=1, q=1, confidence_level=confidence_level)
        #st.write(model_results.summary())
        var_list.extend(['GARCH VaR', 'GARCH VaR - Negative Returns'])
        error_list.extend(['GARCH VaR - Total Percentage Error', 'GARCH VaR - 365D Rolling Percentage Error'])
        #st.write(returns)

    returns, fig_percentage_error, fig_go = risk_functions.plot_var(returns, var_list=var_list, error_list=error_list)
    st.plotly_chart(fig_go, use_container_width=True)
    st.plotly_chart(fig_percentage_error, use_container_width=True)

        



def main():
    st.sidebar.title('Risk Analysis Dashboard')
    st.sidebar.markdown('---')
    menu_list=['Home', 'Ticker Info', 'VaR Model Analysis']
    choice = st.sidebar.radio('Window', menu_list)

    if choice=='Home':
        home()
    if choice=='Ticker Info':
        ticker_info()
    if choice=='VaR Model Analysis':
        model_comparison()


main()