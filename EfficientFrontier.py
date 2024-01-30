from stat import ST_DEV
from statistics import correlation
import yfinance as yahooFinance
import numpy as np
import matplotlib.pyplot as plt 
import mplcursors
import scipy.optimize as optimize



#Returns the Adjusted Closes of all stocks requested
def fetch_stock_data(symbols, start, end):
    
    df = yahooFinance.download(symbols, start, end)["Adj Close"] #Use YahooFinance to pull adjusted closes
    df = df[symbols] #loads the symbols into df
    return df        

def plot_markowitz(weight_arr, port_returns, port_std, sym):
    plt.figure(figsize = (12,6))
    plt.scatter(port_std,port_returns,c = (port_returns / port_std), marker='o')
   
# Enable cursor hover annotations
    weight_str = []
    weights = ""
    for i in weight_arr:
        for j in range(len(i)):
            weights = weights + (sym[j] + " " + "{:.2f}".format(i[j]) + " ")
        weight_str.append(weights)
        weights = ""
    cursor = mplcursors.cursor(hover=True)

# Define the hover function
    def on_hover(sel):
        index = sel.target.index
        x, y = sel.target
        weight = weight_str[index]
        sel.annotation.set_text(weight)

# Connect the hover function to the cursor
    cursor.connect("add", on_hover)
    plt.xlabel('Portfolio Volatility')
    plt.ylabel('Portfolio Return')
    plt.colorbar(label = 'Sharpe ratio (not adjusted for short rate)')
    plt.show()

#Calculate and return the daily returns for each stock
def calculate_daily_returns(stock_data):
    i = len(stock_data)
    daily_returns = np.log(1+stock_data.iloc[:, :i].pct_change()) #Calculates percent change from day to day
    return daily_returns
            
#Generates random weights for each stock in the portfolio
def get_random_weights(df):
  weights = np.random.random(len(df.columns)) #Generate random numbers for weights
  weights /= weights.sum()                    #Normalizes weights
  return weights

#Used to Generate our portfolio(Currently Generates random portfolios)
def gen_portfolio(df, sym):
    daily_returns = calculate_daily_returns(df) 

    port_returns = []
    port_std = []
    weight_arr = []
    for i in range(1000):
        weights = get_random_weights(df)
        weights = np.array(weights)
        weight_arr.append(weights)
        returns = np.sum(daily_returns.mean() * weights) * 252
        stdev = np.sqrt(np.dot(np.dot(daily_returns.cov() * 252, weights), weights.T))
        port_returns.append(returns)
        port_std.append(stdev)
    
    weight_arr = np.array(weight_arr)
    port_returns = np.array(port_returns)
    port_std = np.array(port_std)

    return port_returns, port_std, weight_arr

def minimize_sharpe(weights, returns):
    return -gen_portfolio(weights)['sharpe']

def portfolio_stats(weights, returns):
    returns
    weights = np.array(weights)
    port_return = np.sum(returns.mean() * weights) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe = port_return/port_vol

    return {'return': port_return, 'volatility': port_vol, 'sharpe': sharpe}

def gen_frontier(num_assets):
    constraints = ({'type' : 'eq', 'fun': lambda x: np.sum(x) -1})
    bounds = tuple((0,1) for x in range(num_assets))
    initializer = num_assets * [1./num_assets,]
    print(initializer)
    print(bounds)
    optimal_sharpe=optimize.minimize(minimize_sharpe,
                                 initializer,
                                 method = 'SLSQP',
                                 bounds = bounds,
                                 constraints = constraints)
    print(optimal_sharpe)  


def main():

    symbols = ["ACN", "GOOGL", "ALLY"]
    # Date range
    start = '2023-01-01'
    end = '2024-01-01'

    stock_data = fetch_stock_data(symbols, start, end)
    returns, stdev, weights = gen_portfolio(stock_data, symbols)
    plot_markowitz(weights, returns, stdev, symbols)
   # gen_frontier(len(symbols))

    


if __name__ == "__main__":
    main()