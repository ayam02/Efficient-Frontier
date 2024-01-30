import random
from stat import ST_DEV
from statistics import correlation
import yfinance as yahooFinance
import numpy as np
import pandas as pd
#import pandas_datareader as pdr
import matplotlib.pyplot as plt 
import mplcursors
import scipy.optimize as optimize

#Returns the Adjusted Closes of all stocks requested
def fetch_stock_data(symbols, start, end):
    
    df = yahooFinance.download(symbols, start, end)["Adj Close"] #Use YahooFinance to pull adjusted closes
    df = df[symbols] #loads the symbols into df
    return df