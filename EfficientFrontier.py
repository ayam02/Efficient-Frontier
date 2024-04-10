'''
pip installs required:
- yfinance
- numpy
- mplcursors
- scipy
'''
from stat import ST_DEV
from statistics import correlation
import yfinance as yahooFinance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import mplcursors
import scipy.optimize as optimize

import threading
from OptimizeTarget import OptimizeTarget
import PortfolioStats



'''
Function Name: fetch_stock_data
Inputs: symbols - Array list of tickers
        start - start date of data collection
        end - end date for stock data collection
Returns: df - array of Adjusted Closes of all stocks requested between the times specified
Desc: Accepts an array of symbols and uses yahoo finance to get adjusted closes between start and end date
'''
def fetch_stock_data(symbols, start, end):
    
    df = yahooFinance.download(symbols, start, end)["Adj Close"] #Use YahooFinance to pull adjusted closes
    df = df[symbols]                                             #loads the symbols into df
    return df                                                    #return df array
#end fetch_stock_data
'''+

Function Name: plot_portfolios
Inputs: weight_arr - an list of all arrays of all weightings for each portfolio
        port_returns - list of all portfolios returns
        port_std - list of all portfolio standard deviations
        sym - list of the symbols which they correlate to
Returns: N/A
Desc: Accepts all information about each portfolio that has been created, then plots them on graph
      and enables hover feature for mouse hovering over each portfolio(so weights for that specified
      portfolio returns are visible). Displays Graph
'''
def plot_portfolios(sym, opt_weights, opt_vol, opt_returns):
    plt.figure(figsize = (12,6))                             # Set size of graph
    
    weights = ""
    opt_weight_str = []
    opt_weights = opt_weights *100
    for i in opt_weights:         # Runs for every array of weights in the weight list
        for j in range(len(i)):  # Runs for the amount of weights there are in the array
            weights = weights + (sym[j] + " " + "{:.2f}".format(i[j]) + " ")  # Formats a label which is readable
            #end for
        opt_weight_str.append(weights) # Adds the label to the new Array of labels
        weights = ""               # Empty storage string
        #end for
    opt_points = plt.scatter(opt_vol, opt_returns, c = (opt_returns/opt_vol), marker='x')
    cursor = mplcursors.cursor(hover=True) # Create cursor with hover functionability

    # Define the hover function so label is viewable when hovered over
    def on_hover(sel):
        index = sel.target.index
        x, y = sel.target
        weight = opt_weight_str[index]
        sel.annotation.set_text(weight)

    # Connect the hover function to the cursor
    cursor.connect("add", on_hover)
    
    plt.xlabel('Portfolio Volatility') # Label Axis, and create bar to display Sharpe ratio colorization
    plt.ylabel('Portfolio Return')
    plt.colorbar(label = 'Sharpe ratio (not adjusted for short rate)')
    plt.show()  # Display the graph
#end plot_portfolios

'''
Function Name: calculate_daily_returns
Inputs: stock_data - list of arrays containing daily Adjusted Closes
Returns: daily_returns - list of arrays of the returns on a daily basis from adjusted close of the 
         day prior to that current day
Desc: Calculate and return the daily returns for each stock
'''
def calculate_daily_returns(stock_data):
    #i = len(stock_data)                 # Store amount of stocks we have, for knowledge of the amount of rows
    daily_returns = (1+stock_data.pct_change()) #Calculates percent change from day to day
    return daily_returns
#end calculate_daily_returns


'''
Function Name: minimize_sharpe
Inputs: weights - array of weights
Returns: Negative portfolios stats at weights sharpe ratio
Desc: takes in weights, and returns the negative value of the sharpe ratio
      at those weights for the portfolio corresponding to that.
'''
def minimize_sharpe(weights):
    return -portfolioStats.portfolio_stats(weights)['sharpe']
#end minimize_sharpe


def minimize_ret(weights):
    ret = portfolioStats.portfolio_stats(weights)['return']
    return ret

def maximize_ret(weights):
    returns = pow(np.prod(daily_ret), 252/len(daily_ret)) - 1 # Average annualised returns for each stock
    ret = returns.max() - portfolioStats.portfolio_stats(weights)['return']
    return ret


def min_ret_thread(initializer, bounds, constraints, result):
    min_ret=optimize.minimize(minimize_ret, #the function we are trying to minimize
                            initializer,       #set the starting bounds
                            method = 'SLSQP',  #the method that will be used for optimization
                            bounds = bounds,   #set the max and min value of each weight
                            constraints = constraints) #sets the constraints for which the minimize_vol will be minimized
    result.append(min_ret)

'''
Function Name: get_frontier
Inputs: num_assets - the number of stocks bounds - 
Returns: list of arrays of optimal weights
Desc: gets the returns and then runs an optimization algorithm in order to
      get minimal volatility associated with each level of returns between
      the max returns and min returns
'''
def gen_frontier(num_assets, bounds):
    returns = pow(np.prod(daily_ret), 252/len(daily_ret)) - 1 # Average annualised returns for each stock
    
    initializer = num_assets * [1./num_assets,] #set even initial weight for each asset
    constraints = ({'type':'eq','fun': lambda x: np.sum(x) - 1})

    min_thread_result = []
    min_thread = threading.Thread(target=min_ret_thread, args=(initializer, bounds, constraints, min_thread_result)) 
    min_thread.start()
    max_ret=optimize.minimize(maximize_ret, #the function we are trying to minimize
                                 initializer,       #set the starting bounds
                                 method = 'SLSQP',  #the method that will be used for optimization
                                 bounds = bounds,   #set the max and min value of each weight
                                 constraints = constraints) #sets the constraints for which the minimize_vol will be minimized
    
    max_ret = returns.max() - max_ret['fun']
    min_thread.join()
    min_result = min_thread_result[0]
    min_ret = min_result['fun']
    target_ret = np.linspace(min_ret, max_ret, 50) #create 50 points between the max and min values
    weights = [] #create storage for weights

    threads = []
    min_vol_result = []
    lock = threading.Lock()

    optimizer = OptimizeTarget(initializer,bounds,lock, portfolioStats, daily_ret)
    for t in target_ret: #runs for every point in the 50 targets
        #create a function to find when the minimize function at weights is equal to the target
        thread = threading.Thread(target=optimizer.target_vol_thread, args=(t, min_vol_result))
        threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    for result in min_vol_result:
        weights.append(result['x'])
            
    return np.round(np.array(weights), 8) #round and return the weights
#end gen_frontier


'''
Function Name: get_optimals
Inputs: weights - array of the optimal weights
Outputs: stdevs - array of the standard deviations of the optimal portfolios
         return_arr - array of the returns of the optimal portfolios
Desc: takes in the optimal weights and generates the standard deviations and
      returns for each of those portfolios.
'''
def get_optimals(weights):
    returns = pow(np.prod(daily_ret), 252/len(daily_ret)) - 1 # Average annualised returns for each stock

    return_arr = [] #create storage arrays
    stdevs = []
    for weight in weights: #run for every portfolio
        return_arr.append(np.sum(np.dot(returns, weight))) #store the portfolio returns
        stdevs.append(np.sqrt(np.dot(weight.T, np.dot(daily_ret.cov() * 252, weight)))) #store the portfolio Standard deviation
    return np.array(stdevs), np.array(return_arr)
#end get_optimals

def get_weights(symbols, min_weight, max_weight):
    weight_bounds = list((min_weight, max_weight) for x in range(len(symbols)))
    for i in range(len(symbols)):
        while True:
            try:
                c = int(input("Set Risk Tier for " + symbols[i] + ": "))
                if c == 1:
                    weight_bounds[i] = (0.02, 0.03)
                elif c == 2:
                    weight_bounds[i] = (0.03, 0.04)
                elif c == 3:
                    weight_bounds[i] = (0.04, 1.00)
                else:
                    print("Invalid Risk Tier")
                    continue
                break
            except ValueError:
                print("Please enter a valid integer.")
            

            
    c = input("change max and min weights, input y to change ")
    if(c == "y"):
        for i in range(len(symbols)):
            c = input("change max and min weights for " + symbols[i] + "? input y to change ")
            if(c == "y"):
                min_weight = input("Put Min ")
                max_weight = input("Put max ")
                weight_bounds[i] = (min_weight, max_weight)

    return tuple(weight_bounds)

'''
Function Name: main
Desc: Main method, creates stock list, start and end dates, gets stock data, generates random portfolios
      and plots them.
'''
def main():

    symbols = ["MODG","LVMHF","PEP","JWEL","GIS","TJX","COST","ATD","V","ALLY","SCHW",
               "JPM","BMO","ACN","CSCO","OTEX","DIS","PFE","VRTX","NVO","BEP","ENB","NEE","CNQ","J","MG","XYL","CP","NTR"] # Add tickers you would like to be in the portfolios
    # Date range
    min_weight = 0.02
    max_weight = 0.5
    
    start = '2014-01-01'
    end = '2024-01-01'

    stock_data = fetch_stock_data(symbols, start, end)
    
    global daily_ret
    daily_ret = calculate_daily_returns(stock_data)
    global portfolioStats 
    portfolioStats= PortfolioStats.PortfolioStats(daily_ret)
    weight_bounds = get_weights(symbols, min_weight, max_weight)
    
    opt_weights = gen_frontier(len(symbols), weight_bounds)         #get the optimal weights
    optimal_vol, opt_rets = get_optimals(opt_weights) #get the optimal volatilities and returns
    plot_portfolios(symbols, opt_weights, optimal_vol, opt_rets)            # plot the portfolios on the graph

#end main
    

#run main method
if __name__ == "__main__":
    main()