import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import talib

# User Interface
st.title('Stock Screener Dashboard')

# Date range select
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')

# Ticker data cleaning
ticker_list = pd.read_csv('C:/Users/andre/tickers.csv')
ticker_list = ticker_list.iloc[1:, :]  # Select all rows except the first row
ticker_list = ticker_list[ticker_list.iloc[:, 2].str.contains("NASDAQ|NYSE")]
ticker_list = ticker_list[~ticker_list.iloc[:, 3].str.contains("ETF")]
ticker_list.reset_index(drop=True, inplace=True)
ticker_list = ticker_list.sample(frac=1, random_state=42).reset_index(drop=True) # randomize ticker_list
ticker_list = ticker_list.iloc[1:200, 0] # Number of tickers for calculation ([1:, 0] for all)
ticker_list = ticker_list[ticker_list.str.len() <= 4]
ticker = st.sidebar.selectbox('Select Stock', ticker_list)


# Search for rising volatile stocks
calculate_consistency = st.sidebar.button('Search: Rising Volatile Stocks')
if calculate_consistency:
    progress_bar = st.progress(0)
    progress_text = st.empty()

    consistency_dict = {}
    for i, ticker in enumerate(ticker_list):
        data = yf.download(ticker, start=start_date, end=end_date)
        if not data.empty:
            data['Return'] = data['Adj Close'].pct_change()
            weekly_increase = data['Adj Close'].resample('W').ffill().pct_change().mean() * 100  # Calculate weekly increase
            volatility = data['Return'].std() * 100  # Calculate volatility

            # Calculate moving averages
            short_term_ma = talib.SMA(data['Adj Close'], timeperiod=20)
            mid_term_ma = talib.SMA(data['Adj Close'], timeperiod=50)
            long_term_ma = talib.SMA(data['Adj Close'], timeperiod=200)

            # Assess bullish motion using moving averages
            short_term_score = ((short_term_ma[-1] - short_term_ma[-2]) / short_term_ma[-2])*50
            mid_term_score = ((mid_term_ma[-1] - mid_term_ma[-2]) / mid_term_ma[-2])*50
            long_term_score = ((long_term_ma[-1] - long_term_ma[-2]) / long_term_ma[-2])*50

            # Calculate consistency score
            distance = abs(volatility - 4.0)
            weight = 1 - distance / 4.0  # Increase score if volatility is around 4.0
            rising_score = ((volatility * weight) / 3) + weekly_increase + short_term_score + mid_term_score + long_term_score

            consistency_dict[ticker] = (weekly_increase, volatility, short_term_score, mid_term_score, long_term_score, rising_score)

        # Update progress bar and text
        progress_bar.progress((i + 1) / len(ticker_list))
        progress_text.text(f'Calculating {i + 1} of {len(ticker_list)}')

    # Sort tickers by consistency score in descending order
    sorted_tickers = sorted(consistency_dict, key=lambda x: consistency_dict[x][5] if not np.isnan(consistency_dict[x][5]) else float('-inf'), reverse=True)

    # Display ranked tickers with weekly_increase, volatility, short-term score, mid-term score, long-term score, and rising_score
    st.sidebar.markdown('### Stock List (Ranked by Rising Score)')
    for rank, ticker in enumerate(sorted_tickers, 1):
        weekly_increase, volatility, short_term_score, mid_term_score, long_term_score, rising_score = consistency_dict[ticker]

        # Set colors based on values
        weekly_increase_color = 'green' if weekly_increase > 2.0 else 'red' if weekly_increase < 0 else 'yellow'
        volatility_color = 'green' if 2.5 < volatility < 6.5 else 'yellow'
        short_term_color = 'green' if short_term_score > 0 else 'red'
        mid_term_color = 'green' if mid_term_score > 0 else 'red'
        long_term_color = 'green' if long_term_score > 0 else 'red'
        rising_score_color = 'pink'

        # Format the text with colors and percentages
        formatted_text = '{}. <span style="font-size: 20px; font-weight: bold">{}</span><br>&emsp;Weekly Increase: <span style="color:{}">{:.2f}%</span><br>&emsp;Volatility: <span style="color:{}">{:.2f}%</span><br>&emsp;Short: <span style="color:{}">{:.4f}</span><br>&emsp;Mid: <span style="color:{}">{:.4f}</span><br>&emsp;Long: <span style="color:{}">{:.4f}</span><br>&emsp;Rising Score: <span style="color:{}">{:.4f}</span>'.format(
            rank, ticker, weekly_increase_color, weekly_increase, volatility_color, volatility,
            short_term_color, short_term_score, mid_term_color, mid_term_score,
            long_term_color, long_term_score, rising_score_color, rising_score
        )
        formatted_text = '<pre>{}</pre>'.format(formatted_text)
        st.sidebar.markdown(formatted_text, unsafe_allow_html=True)


# Search for consistant long term stocks with high annual return
calculate_consistency = st.sidebar.button('Search: Consistant Long Term')
if calculate_consistency:
    progress_bar = st.progress(0)
    progress_text = st.empty()

    consistency_dict = {}
    for i, ticker in enumerate(ticker_list):
        data = yf.download(ticker, start=start_date, end=end_date)
        if not data.empty:
            data['Return'] = data['Adj Close'].pct_change()
            weekly_increase = data['Adj Close'].resample('W').ffill().pct_change().mean() * 100  # Calculate weekly increase
            volatility = data['Return'].std() * 100  # Calculate volatility

            # Calculate moving averages
            short_term_ma = talib.SMA(data['Adj Close'], timeperiod=20)
            mid_term_ma = talib.SMA(data['Adj Close'], timeperiod=50)
            long_term_ma = talib.SMA(data['Adj Close'], timeperiod=200)

            # Assess bullish motion using moving averages
            short_term_score = ((short_term_ma[-1] - short_term_ma[-2]) / short_term_ma[-2])*50
            mid_term_score = ((mid_term_ma[-1] - mid_term_ma[-2]) / mid_term_ma[-2])*50
            long_term_score = ((long_term_ma[-1] - long_term_ma[-2]) / long_term_ma[-2])*350

            # Calculate consistency score
            distance = abs(volatility - 1.5)
            volatility_score = 1 - (distance / 1.5)
            rising_score = volatility_score + weekly_increase + short_term_score + mid_term_score + long_term_score

            consistency_dict[ticker] = (weekly_increase, volatility, short_term_score, mid_term_score, long_term_score, rising_score)

        # Update progress bar and text
        progress_bar.progress((i + 1) / len(ticker_list))
        progress_text.text(f'Calculating {i + 1} of {len(ticker_list)}')

    # Sort tickers by consistency score in descending order
    sorted_tickers = sorted(consistency_dict, key=lambda x: consistency_dict[x][5] if not np.isnan(consistency_dict[x][5]) else float('-inf'), reverse=True)

    # Display ranked tickers with weekly_increase, volatility, short-term score, mid-term score, long-term score, and rising_score
    st.sidebar.markdown('### Stock List (Ranked by Rising Score)')
    for rank, ticker in enumerate(sorted_tickers, 1):
        weekly_increase, volatility, short_term_score, mid_term_score, long_term_score, rising_score = consistency_dict[ticker]

        # Set colors based on values
        weekly_increase_color = 'green' if weekly_increase > 2.0 else 'red' if weekly_increase < 0 else 'yellow'
        volatility_color = 'green' if volatility > 2.5 else 'yellow'
        short_term_color = 'green' if short_term_score > 0 else 'red'
        mid_term_color = 'green' if mid_term_score > 0 else 'red'
        long_term_color = 'green' if long_term_score > 0 else 'red'
        rising_score_color = 'pink'

        # Format the text with colors and percentages
        formatted_text = '{}. <span style="font-size: 20px; font-weight: bold">{}</span><br>&emsp;Weekly Increase: <span style="color:{}">{:.2f}%</span><br>&emsp;Volatility: <span style="color:{}">{:.2f}%</span><br>&emsp;Short: <span style="color:{}">{:.4f}</span><br>&emsp;Mid: <span style="color:{}">{:.4f}</span><br>&emsp;Long: <span style="color:{}">{:.4f}</span><br>&emsp;Rising Score: <span style="color:{}">{:.4f}</span>'.format(
            rank, ticker, weekly_increase_color, weekly_increase, volatility_color, volatility,
            short_term_color, short_term_score, mid_term_color, mid_term_score,
            long_term_color, long_term_score, rising_score_color, rising_score
        )
        formatted_text = '<pre>{}</pre>'.format(formatted_text)
        st.sidebar.markdown(formatted_text, unsafe_allow_html=True)


# Search for sideways volatile stocks
calculate_consistency = st.sidebar.button('Search: Sideways Volatile Stocks')
if calculate_consistency:
    progress_bar = st.progress(0)
    progress_text = st.empty()

    consistency_dict = {}
    for i, ticker in enumerate(ticker_list):
        data = yf.download(ticker, start=start_date, end=end_date)
        if not data.empty:
            data['Return'] = data['Adj Close'].pct_change()
            weekly_increase = data['Adj Close'].resample('W').ffill().pct_change().mean() * 100  # Calculate weekly increase
            volatility = data['Return'].std() * 100  # Calculate volatility

            # Calculate moving averages
            short_term_ma = talib.SMA(data['Adj Close'], timeperiod=20)
            mid_term_ma = talib.SMA(data['Adj Close'], timeperiod=50)
            long_term_ma = talib.SMA(data['Adj Close'], timeperiod=200)

            # Assess bullish motion using moving averages
            short_term_score = ((short_term_ma[-1] - short_term_ma[-2]) / short_term_ma[-2])*50
            mid_term_score = ((mid_term_ma[-1] - mid_term_ma[-2]) / mid_term_ma[-2])*50
            long_term_score = ((long_term_ma[-1] - long_term_ma[-2]) / long_term_ma[-2])*50

            # Calculate consistency score
            distance = abs(volatility - 4.0)
            weight = 1 - distance / 4.0  # Increase score if volatility is around 4.0
            rising_score = ((volatility * weight) / 3) + weekly_increase + short_term_score + mid_term_score + long_term_score

            consistency_dict[ticker] = (weekly_increase, volatility, short_term_score, mid_term_score, long_term_score, rising_score)

        # Update progress bar and text
        progress_bar.progress((i + 1) / len(ticker_list))
        progress_text.text(f'Calculating {i + 1} of {len(ticker_list)}')

    # Sort tickers by consistency score in descending order
    sorted_tickers = sorted(consistency_dict, key=lambda x: consistency_dict[x][5] if not np.isnan(consistency_dict[x][5]) else float('-inf'), reverse=True)

    # Display ranked tickers with weekly_increase, volatility, short-term score, mid-term score, long-term score, and rising_score
    st.sidebar.markdown('### Stock List (Ranked by Rising Score)')
    for rank, ticker in enumerate(sorted_tickers, 1):
        weekly_increase, volatility, short_term_score, mid_term_score, long_term_score, rising_score = consistency_dict[ticker]

        # Set colors based on values
        weekly_increase_color = 'green' if weekly_increase > 2.0 else 'red' if weekly_increase < 0 else 'yellow'
        volatility_color = 'green' if 2.5 < volatility < 6.5 else 'yellow'
        short_term_color = 'green' if short_term_score > 0 else 'red'
        mid_term_color = 'green' if mid_term_score > 0 else 'red'
        long_term_color = 'green' if long_term_score > 0 else 'red'
        rising_score_color = 'pink'

        # Format the text with colors and percentages
        formatted_text = '{}. <span style="font-size: 20px; font-weight: bold">{}</span><br>&emsp;Weekly Increase: <span style="color:{}">{:.2f}%</span><br>&emsp;Volatility: <span style="color:{}">{:.2f}%</span><br>&emsp;Short: <span style="color:{}">{:.4f}</span><br>&emsp;Mid: <span style="color:{}">{:.4f}</span><br>&emsp;Long: <span style="color:{}">{:.4f}</span><br>&emsp;Rising Score: <span style="color:{}">{:.4f}</span>'.format(
            rank, ticker, weekly_increase_color, weekly_increase, volatility_color, volatility,
            short_term_color, short_term_score, mid_term_color, mid_term_score,
            long_term_color, long_term_score, rising_score_color, rising_score
        )
        formatted_text = '<pre>{}</pre>'.format(formatted_text)
        st.sidebar.markdown(formatted_text, unsafe_allow_html=True)

# Search (stocks that go well for a particular stratagy) 
# search stratagy = st.sidebar.button('')
        
# Show stocks on yahoo finance
data = yf.download(ticker,start=start_date, end=end_date)
if data.empty:
    st.error("No data available for the specified ticker and date range.")
else:
    fig = px.line(data, x=data.index, y = data['Adj Close'], title = ticker)
    st.plotly_chart(fig)




# Pricing Information Tabs
pricing_data, fundamental_data, news, tech_indicator = st.tabs(["Pricing Data", "Fundamental Data", "Top 10 News", "Technical Analysis Dashboard"])

with pricing_data:
    st.header('Price Movements')
    data2 = data
    data2['% Change'] = data['Adj Close'] / data['Adj Close'].shift(1) - 1
    data2.dropna(inplace = True)
    st.write(data2)
    annual_return = data2['% Change'].mean()*252*100
    st.write('Annual Return is ',annual_return,'%')
    stdev = np.std(data2['% Change'])*np.sqrt(252)
    st.write('Standard Deviation is ', stdev*100,'%')
    st.write('Risk Adj. Return is ', annual_return/(stdev*100))
    
from alpha_vantage.fundamentaldata import FundamentalData 
with fundamental_data:
    key = 'LI7IYZMP8JT42X6J'
    fd = FundamentalData(key,output_format = 'pandas')
    st.subheader('Balance Sheet')
    balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
    bs = balance_sheet.T[2:]
    bs.columns = list(balance_sheet.T.iloc[0])
    st.write(bs)
    st.subheader('Income Statement')
    income_statement = fd.get_income_statement_annual(ticker)[0]
    is1 = income_statement.T[2:]
    is1.columns = list(income_statement.T.iloc[0])
    st.write(is1)
    st.subheader('Cash Flow Statement')
    cash_flow = fd.get_cash_flow_annual(ticker)[0]
    cf = cash_flow.T[2:]
    cf.columns = list(cash_flow.T.iloc[0])
    st.write(cf)   

from stocknews import StockNews
with news:
    st.header(f'News of {ticker}')
    sn = StockNews(ticker, save_news = False)
    df_news = sn.read_rss()
    for i in range(10): # Top 10 news
        st.subheader(f'news {i+1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment = df_news['sentiment_title'][i]
        st.write(f'Title Sentiment {title_sentiment}')
        news_sentiment = df_news['sentiment_summary'][i]
        st.write(f'News Sentiment {news_sentiment}')

import pandas_ta as ta
with tech_indicator:
    st.subheader('Technical Analasys Dashboard:')
    df = pd.DataFrame()
    ind_list = df.ta.indicators(as_list=True)
    technical_indicator = st.selectbox('Tech Indicator', options=ind_list)
    method = technical_indicator
    indicator = pd.DataFrame(getattr(ta,method)(low=data["Low"], close=data["Close"], high=data["High"], open=data["Open"], volume=data["Volume"]))
    indicator["Close"] = data["Close"]
    figW_ind_new = px.line(indicator)
    st.plotly_chart(figW_ind_new)
    st.write(indicator)