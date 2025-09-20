# MovementBasedStockScreener
(Edit: This method was orignally built to scan through each ticker for any given criteria built into the program. in this case it was used to find stocks which had a steady but quick uprising in price. along other options to find repeating volatile patterns. Its built compltely on python and is visually shown on a webpage through streamlit. The price data is from alpha vantage API)

This is a streamlit program that scans the stock market one by one for stocks that follow a certain movement. This is useful for finding stocks that work well with specific strategies.

This program requires a csv file of all of the tickers in the stock market. Some tickers may not work if they arent supported by alpha vantage which could cause unnecessary computation time.

