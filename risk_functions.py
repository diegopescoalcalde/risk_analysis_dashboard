import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import shapiro, norm
import arch
from prophet import Prophet


def compute_returns(tickers, dict_tickers):
  yf_data = yf.download(dict_tickers[tickers], period='5y', interval='1d')
  yf_data.columns = yf_data.columns.droplevel(1)
  returns = yf_data[['Close']].pct_change().dropna()
  returns.columns = ['Returns']

  # Calculate normal distribution with same mean and std as the returns
  returns['Normal Distribution'] = np.random.normal(loc=returns['Returns'].mean(), scale=returns['Returns'].std(), size=len(returns))

  # Candlestick chart
  fig_candlesticks = go.Figure(data=[go.Candlestick(x=yf_data.index,
                                            open=yf_data['Open'],
                                            high=yf_data['High'],
                                            low=yf_data['Low'],
                                            close=yf_data['Close'],
                                            increasing_line_color= 'green', 
                                            decreasing_line_color= 'red')])
  fig_candlesticks.update_layout(xaxis_title='Date',
                yaxis_title='Price',
                title=tickers, 
                xaxis_rangeslider_visible=False)


  # Line plot returns
  fig_returns_line = px.line(returns,
            x=returns.index,
            y=['Returns'],
            color_discrete_sequence=px.colors.qualitative.G10)

  fig_returns_line.update_layout(title=f"{tickers} Daily Returns",
      xaxis_title="Date",
      yaxis_title="Returns",
      legend=dict(title='Legend',
            orientation="h",
            yanchor="bottom",
            y=-0.7,
            xanchor="left",
            x=0.01
            ))

  # Histogram of returns
  fig_returns_hist = px.histogram(returns,
            x=['Returns', 'Normal Distribution'],
                    barmode='overlay',
                    color_discrete_sequence=px.colors.qualitative.G10)

  fig_returns_hist.update_layout(title=f"{tickers} Returns Distribution",
      xaxis_title="Returns",
      autosize=True,
      legend=dict(title='Legend',
            orientation="h",
            yanchor="bottom",
            y=-0.7,
            xanchor="left",
            x=0.01
            ))
  
  # Shapiro-Wilk normality test
  normality_test = shapiro(returns)


  return returns, fig_candlesticks, fig_returns_line, fig_returns_hist, normality_test



#------------------------------------------------------------------------------------



def calculate_historical_var(df_returns, window, confidence_level, min_periods=1):
  risk_percentile = (1-confidence_level)/2
  df_returns['Historical VaR'] = df_returns['Returns'].rolling(window=window, min_periods=min_periods).quantile(1-risk_percentile).shift(1)
  df_returns['Historical VaR - Negative Returns'] = df_returns['Returns'].rolling(window=window, min_periods=min_periods).quantile(risk_percentile).shift(1)
  # Calculate percentage error
  df_returns['Actual > Predicted'] = (df_returns['Returns'] > df_returns['Historical VaR']) | (df_returns['Returns'] < df_returns['Historical VaR - Negative Returns'])
  df_returns['Historical VaR - Total Percentage Error'] = df_returns['Actual > Predicted'].expanding().sum()*100 / df_returns['Actual > Predicted'].expanding().count()
  df_returns['Historical VaR - 365D Rolling Percentage Error'] = df_returns['Actual > Predicted'].rolling('365D').sum()*100 / df_returns['Actual > Predicted'].rolling('365D').count()
  
  return df_returns



#------------------------------------------------------------------------------------



def compute_ewma_var(returns, lambda_value, time_horizon, confidence_level):
  risk_percentile = 1-((1-confidence_level)/2)

  # Calculate squared returns
  returns['Squared Returns'] = returns['Returns']**2

  # Calculate EWMA volatility
  returns['EWMA Volatility'] = pd.Series(returns['Squared Returns']).ewm(alpha=1-lambda_value, adjust=False).mean().pow(0.5).shift(1)
  returns['Negative EWMA Volatility'] = -returns['EWMA Volatility']

  # Calculate the critical value for the standard normal distribution
  z_score_critical_value = norm.ppf(risk_percentile)

  # Calculate VaR
  returns['EWMA VaR'] = returns['EWMA Volatility'] * z_score_critical_value * np.sqrt(time_horizon)
  returns['EWMA VaR - Negative Returns'] = -returns['EWMA VaR']

  # Calculate percentage error
  returns['Actual > Predicted'] = returns['Returns'].abs() > returns['EWMA VaR']
  returns['EWMA VaR - Total Percentage Error'] = returns['Actual > Predicted'].expanding().sum()*100 / returns['Actual > Predicted'].expanding().count()
  returns['EWMA VaR - 365D Rolling Percentage Error'] = returns['Actual > Predicted'].rolling('365D').sum()*100 / returns['Actual > Predicted'].rolling('365D').count()

  return returns



#------------------------------------------------------------------------------------



def compute_garch_var(returns, p, q, confidence_level):
   # Fit model
   model = arch.arch_model(returns['Returns'], mean='Constant', vol='GARCH', p=p, q=q)
   model_results = model.fit()
   
   # Calculate VaR
   risk_percentile = 1-((1-confidence_level)/2)
   percentile = model.distribution.ppf([risk_percentile])
   forecast = model_results.forecast(start=returns.index[0])
   cond_mean = forecast.mean[:]
   cond_variance = forecast.variance[:]
   garch_var =  cond_mean.values + np.sqrt(cond_variance).values * percentile
   garch_var = pd.DataFrame(garch_var, index=returns.index)
   returns['GARCH VaR'] = garch_var.shift(1)
   returns['GARCH VaR - Negative Returns'] = -garch_var.shift(1)

   # Calculate percentage error
   returns['Actual > Predicted'] = returns['Returns'].abs() > returns['GARCH VaR']
   returns['GARCH VaR - Total Percentage Error'] = returns['Actual > Predicted'].expanding().sum()*100 / returns['Actual > Predicted'].expanding().count()
   returns['GARCH VaR - 365D Rolling Percentage Error'] = returns['Actual > Predicted'].rolling('365D').sum()*100 / returns['Actual > Predicted'].rolling('365D').count()

   return model_results, garch_var, returns



#------------------------------------------------------------------------------------



def plot_var(returns, var_list, error_list):
  
  colors=px.colors.qualitative.G10

  fig_go = go.Figure()
  fig_go.add_trace(go.Scatter(
            x=returns.index,
            y=returns['Returns'],
            name='Returns',
            line=dict(color=colors[0])
                ))
  for i, var in enumerate(var_list):
    if var in returns.columns:
        fig_go.add_trace(go.Scatter(
                    x=returns.index,
                    y=returns[var],
                    name=var,
                    line=dict(color=colors[i+1])
                        ))
  
  fig_go.update_layout(title="Predicted VaR x Actual Returns",
      xaxis_title="Date",
      yaxis_title="Returns",
      autosize=True,
      legend=dict(title='Legend',
            orientation="h",
            yanchor="bottom",
            y=-0.7,
            xanchor="left",
            x=0.01
            ))
  
  
  
  fig_percentage_error = go.Figure()
  for i, error_metric in enumerate(error_list):
    if error_metric in returns.columns:
        fig_percentage_error.add_trace(go.Scatter(
                    x=returns.index,
                    y=returns[error_metric],
                    name=error_metric,
                    line=dict(color=colors[i])
                        ))
  
  fig_percentage_error.update_layout(title="Estimation Error Frequency",
      xaxis_title="Date",
      yaxis_title="Error Frequency (%)",
      legend=dict(title='Legend',
            orientation="h",
            yanchor="bottom",
            y=-0.7,
            xanchor="left",
            x=0.01
            ))


  return returns, fig_percentage_error, fig_go



#------------------------------------------------------------------------------------



def anomaly(df, column, interval_width):
    # Create columns ds and y and fit prophet model
    df['ds'] = df.index.copy()
    df['y'] = df[column].copy()
    model = Prophet(interval_width=interval_width, yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df)
    forecast = model.predict(df)

    # Combine existing dataframe with forecasted one
    df = pd.merge(df, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')

    # Identify anomalies in boolean column
    df['Anomaly'] = df.apply(lambda rows: True if ((rows.y<rows.yhat_lower)|(rows.y>rows.yhat_upper)) else False, axis = 1)

    # Set date index
    df.set_index('ds', inplace=True)

    # Create figure
    fig_anomaly = go.Figure()
    fig_anomaly.add_trace(go.Scatter(
        x=df[df['Anomaly'] == False].index,
        y=df[df['Anomaly'] == False][column],
        mode='markers',
        name='Normal',
        line=dict(color='blue')
            ))
    fig_anomaly.add_trace(go.Scatter(
        x=df[df['Anomaly'] == True].index,
        y=df[df['Anomaly'] == True][column],
        mode='markers',
        name='Anomaly',
        line=dict(color='red')
            ))
    fig_anomaly.update_layout(title=f"Anomaly Detection - {column}",
    xaxis_title="Date",
    yaxis_title=column,
    autosize=True,
    legend=dict(title='Legend',
            orientation="h",
            yanchor="bottom",
            y=-0.7,
            xanchor="left",
            x=0.01
            ))
    
    return df, fig_anomaly



#------------------------------------------------------------------------------------



def portfolio_analysis(tickers, dict_tickers):
    tickers_df=None

    for ticker in tickers:
        yf_data = yf.download(dict_tickers[ticker], period='5y', interval='1d')
        yf_data.columns = yf_data.columns.droplevel(1)
        yf_data = yf_data[['Close']].pct_change().dropna()
        yf_data.columns = [ticker]
        if tickers_df is None:
            tickers_df = yf_data.copy()
        else:
            tickers_df = tickers_df.merge(yf_data, left_index=True, right_index=True)
    
    # Define equal weights
    weights = np.ones(shape=len(tickers))
    weights = weights/weights.sum()

    # Calculate Portfolio Returns
    tickers_df['Portfolio'] = tickers_df.mean(axis=1)

    compounded_returns = tickers_df.copy()
    compounded_returns = (1+compounded_returns).cumprod()
    fig_returns = px.line(compounded_returns,
                          x=compounded_returns.index,
                          y=compounded_returns.columns,
                          color_discrete_sequence=px.colors.qualitative.G10)
    
    fig_returns.update_layout(title="Portfolio Normalized Compounded Return",
        xaxis_title="Date",
        yaxis_title="Normalized Compounded Return",
        autosize=True,
        legend=dict(title='Legend',
                orientation="h",
                yanchor="bottom",
                y=-0.7,
                xanchor="left",
                x=0.01
                ))

    # Calculate Portfolio Volatility from returns
    portfolio_volatility = tickers_df.rolling('365D', min_periods=1).std()
        
    fig_volatility = px.line(portfolio_volatility,
                          x=portfolio_volatility.index,
                          y=portfolio_volatility.columns,
                          color_discrete_sequence=px.colors.qualitative.G10)
        
    fig_volatility.update_layout(title="365D Rolling Portfolio Volatility",
        xaxis_title="Date",
        yaxis_title="Volatility",
        autosize=True,
        legend=dict(title='Legend',
                orientation="h",
                yanchor="bottom",
                y=-0.7,
                xanchor="left",
                x=0.01
                ))
    
    weights = pd.DataFrame(weights)
    weights = weights.transpose()
    weights.columns = tickers

    return weights, tickers_df, compounded_returns, fig_volatility, fig_returns
