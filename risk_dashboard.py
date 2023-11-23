import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import shapiro, norm
import risk_functions
from prophet import Prophet



def home():
    st.title('Home')
    st.header('Welcome! Please read me!')
    st.markdown('''
            Thank you for accessing this simple risk analysis dashboard. The objective of this application is to facilitate
            the understanding of basic market risk tools and models, using real data and interactive charts. This is not
            an input for real trading, please don't use it for this purpose. 
                
            Below you can find some useful information about the different sections of this app, which you can
            access in the sidebar menu. Charts are optimized for the desktop version.''')
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
        
    with st.expander('* Portfolio Analysis'):
        st.markdown('''
                    This section calculates the compounded return and volatility of an equal weight portfolio composed
                    of two or more assets. The results help to visualize the main message of Markowitz's Modern Portfolio Theory, 
                    which is that diversification is helpful in reducing volatility while keeping moderate returns 
                    when the selected assets have small values of covariance.
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

                The first chart presents the calculated VaR for each model in every day of the backtested period.
                The calculated VaR is plotted on top of the asset's historic returns so that you can visually
                compare the real returns with the backtested VaR.
                
                The second chart plots the estimation error frequency for each model. This value is closely
                related with the confidence level set, and is the frequency of days which the historical returns
                exceeded the estimated VaR. A well defined model will obtain an estimation error frequency close to 5%
                when the confidence level is set to 95%.
                
            ''')
                
    with st.expander('Anomaly Detection'):
        st.markdown('''
                An anomaly is an unusual or unexpected event or pattern in data. In the financial markets, anomalies
                might include sudden spikes or drops in price, unusual trading volumes, or other events
                that deviate from previous patterns. Anomaly detection is the process of identifying these events
                in data using statistical and mathematical methods. Detected anomalies can trigger many actions
                such as trading operations to improve returns or minimize risks.

                Depending on the model that is used, different parameters must be established, but most of them
                are related to the sensitiveness of the model, i.e. how significant must the deviation be to
                be categorized as an anomaly. In this application that parameter will be the Interval Width,
                which can be interpreted as a confidence level. The higher the confidence level required, the
                lower the quantity of data points flagged as anomalies.
                    ''')
    
    st.markdown('Developed by Diego Pesco Alcalde')
    st.link_button("LinkedIn", "https://www.linkedin.com/in/diegopesco/")



#------------------------------------------------------------------------------------


                
def ticker_info():
    st.title('Ticker Info')

    # Define tickers options and present to user
    tickers_options = ['S&P500', 'NASDAQ', 'USD/BRL', 'Gold', 'BTC/USD', 'MCHI']
    dict_tickers = {
        'S&P500':'^GSPC',
        'NASDAQ':'^IXIC',
        'USD/BRL':'BRL=X',
        'Gold':'GC=F',
        'BTC/USD':'BTC-USD',
        'MCHI':'MCHI'
    }

    tickers = st.selectbox('Choose Ticker', tickers_options)
    st.markdown('Select area with mouse to zoom charts, click twice to zoom out.')
    
    try:
        # Retrieve data and charts from ticker
        returns, fig_candlesticks, fig_returns_line, fig_returns_hist, normality_test = risk_functions.compute_returns(tickers, dict_tickers)

        # Plot charts and results
        st.plotly_chart(fig_candlesticks, use_container_width=True)
        st.plotly_chart(fig_returns_line, use_container_width=True)
        st.plotly_chart(fig_returns_hist, use_container_width=True)
        st.text('Shapiro-Wilk Normality Test Results')
        st.markdown(f'Test Statistic: {round(normality_test[0], 5)}')
        st.markdown(f'P-value: {round(normality_test[1], 4):4f}')
    except Exception as e:
        st.markdown('Please try reloading this page or try another ticker.')
        print(e)



#------------------------------------------------------------------------------------



def portfolio():
    st.title('Portfolio Analysis')

    # Define tickers options and present to user
    tickers_options = ['S&P500', 'NASDAQ', 'USD/BRL', 'Gold', 'BTC/USD', 'MCHI']
    dict_tickers = {
        'S&P500':'^GSPC',
        'NASDAQ':'^IXIC',
        'USD/BRL':'BRL=X',
        'Gold':'GC=F',
        'BTC/USD':'BTC-USD',
        'MCHI':'MCHI'
    }

    tickers = st.multiselect('Choose Tickers to Build Portfolio', tickers_options, 'S&P500')
    
    try:
        # Retrieve data and charts from portfolio
        weights, tickers_df, compounded_returns, fig_volatility, fig_returns = risk_functions.portfolio_analysis(tickers, dict_tickers)
        
        # Plot charts and data
        st.markdown('**Portfolio Weights**')    
        st.dataframe(weights, hide_index=True)
        st.plotly_chart(fig_returns, use_container_width=True)
        st.plotly_chart(fig_volatility, use_container_width=True)
        
    except Exception as e:
        st.markdown('Please try reloading this page or try another ticker.')
        print(e)



#------------------------------------------------------------------------------------



def model_comparison():
    st.title('VaR Model Analysis')

    # Define tickers options and present to user
    tickers_options = ['S&P500', 'NASDAQ', 'USD/BRL', 'Gold', 'BTC/USD', 'MCHI']
    dict_tickers = {
        'S&P500':'^GSPC',
        'NASDAQ':'^IXIC',
        'USD/BRL':'BRL=X',
        'Gold':'GC=F',
        'BTC/USD':'BTC-USD',
        'MCHI':'MCHI'
    }

    tickers = st.selectbox('Choose Ticker', tickers_options)

    try:
        # Retrieve data from ticker
        returns, fig_candlesticks, fig_returns_line, fig_returns_hist, normality_test = risk_functions.compute_returns(tickers, dict_tickers)
        
        # Request user model choice and confidence level
        models = st.multiselect('Select Risk Models', ['Historical VaR', 'EWMA VaR', 'GARCH VaR'], ['EWMA VaR'])
        var_list = []
        error_list = []
        confidence_level = st.number_input('Confidence Level', min_value=0.00, max_value=1.00, value=0.95)

        # Retrieve model results according to the input
        if 'EWMA VaR' in models:
            lambda_value = st.number_input('EWMA VaR - Decay Factor', min_value=0.00, max_value=1.00, value=0.94)
            returns = risk_functions.compute_ewma_var(returns, lambda_value=lambda_value, time_horizon=1, confidence_level=confidence_level)
            var_list.extend(['EWMA VaR', 'EWMA VaR - Negative Returns'])
            error_list.extend(['EWMA VaR - Total Percentage Error', 'EWMA VaR - 365D Rolling Percentage Error'])
        if 'Historical VaR' in models:
            window = st.number_input('Historical VaR - Lookback Window (in Days)', min_value=1, max_value=2000, value=1095)
            returns = risk_functions.calculate_historical_var(returns, window=window, confidence_level=confidence_level)
            var_list.extend(['Historical VaR', 'Historical VaR - Negative Returns'])
            error_list.extend(['Historical VaR - Total Percentage Error', 'Historical VaR - 365D Rolling Percentage Error'])
        if 'GARCH VaR' in models:
            model_results, garch_var, returns = risk_functions.compute_garch_var(returns, p=1, q=1, confidence_level=confidence_level)
            var_list.extend(['GARCH VaR', 'GARCH VaR - Negative Returns'])
            error_list.extend(['GARCH VaR - Total Percentage Error', 'GARCH VaR - 365D Rolling Percentage Error'])

        # Retrieve charts and plot to user
        returns, fig_percentage_error, fig_go = risk_functions.plot_var(returns, var_list=var_list, error_list=error_list)
        st.plotly_chart(fig_go, use_container_width=True)
        st.plotly_chart(fig_percentage_error, use_container_width=True)
    except Exception as e:
        st.markdown('Please try reloading this page or try another ticker.')
        print(e)



#------------------------------------------------------------------------------------



def anomaly_detection():
    st.title('Anomaly Detection')

    # Define tickers options and present to user
    tickers_options = ['S&P500', 'NASDAQ', 'USD/BRL', 'Gold', 'BTC/USD', 'MCHI']
    dict_tickers = {
        'S&P500':'^GSPC',
        'NASDAQ':'^IXIC',
        'USD/BRL':'BRL=X',
        'Gold':'GC=F',
        'BTC/USD':'BTC-USD',
        'MCHI':'MCHI'
    }

    tickers = st.selectbox('Choose Ticker', tickers_options)
    interval_width = st.number_input('Interval Width', min_value=0.00, max_value=1.00, value=0.90)

    try:
        yf_data = yf.download(dict_tickers[tickers], period='5y', interval='1d')
        volume = yf_data[['Volume']].copy()
        returns = yf_data[['Adj Close']].pct_change().dropna()
        returns.columns = ['Returns']

        volume, fig_volume = risk_functions.anomaly(df=volume, column='Volume', interval_width=interval_width)
        st.plotly_chart(fig_volume, use_container_width=True)

        returns, fig_returns = risk_functions.anomaly(df=returns, column='Returns', interval_width=interval_width)
        st.plotly_chart(fig_returns, use_container_width=True)


    except Exception as error:
        st.markdown('Please try reloading this page or try another ticker.')
        print(error)



#------------------------------------------------------------------------------------



def main():
    st.sidebar.title('Risk Analysis Dashboard')
    st.sidebar.markdown('---')
    menu_list=['Home', 'Ticker Info', 'Portfolio Analysis', 'VaR Model Analysis', 'Anomaly Detection']
    choice = st.sidebar.radio('Window', menu_list)

    if choice=='Home':
        home()
    if choice=='Ticker Info':
        ticker_info()
    if choice=='Portfolio Analysis':
        portfolio()
    if choice=='VaR Model Analysis':
        model_comparison()
    if choice=='Anomaly Detection':
        anomaly_detection()


main()