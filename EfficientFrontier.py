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
    return -portfolio_stats(weights)['sharpe']
#end minimize_sharpe

'''
Function Name: minimize_vol
Inputs: weights - array of weights
Returns: the volatility or std of the portfolio at those weights
Desc: takes in weights, run portfolio_stats funtion with those weights
      and returns the volitility associated with that portfolio
'''
def minimize_vol(weights):
    vol = portfolio_stats(weights)['volatility']
    return vol
#end minimize_vol

def minimize_ret(weights):
    ret = portfolio_stats(weights)['return']
    return ret

def maximize_ret(weights):
    returns = calculate_daily_returns(stock_data) # get daily returns
    returns = pow(np.prod(returns), 252/len(returns)) - 1 # Average annualised returns for each stock
    ret = returns.max() - portfolio_stats(weights)['return']
    return ret

'''
Function Name: portfolio_stats
Inputs: weights
Returns: port_return - the return associated with the portfolio
         port_vol - the volatility associated with the portfolio
         sharpe - the sharpe rate associated with the portfolio
Desc: Takes in weights, then calculates and returns the portfolio's
      return, volatility, and sharpe ratio associated with those weights
'''
def portfolio_stats(weights):
    returns= calculate_daily_returns(stock_data) # get daily returns
    weights = np.array(weights)                  # set weights into a numpy array, if it is a list
    port_return = np.sum(np.dot((pow(np.prod(returns), 252/len(returns)) - 1), weights)) #calculate annualized return per stock and multiplys by weights and adds up returns
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights))) # calculates standard deviation of the portfolio
    sharpe = port_return/port_vol # calculates sharpe ratio
    return {'return': port_return, 'volatility': port_vol, 'sharpe': sharpe}
#end portfolio_stats


'''
Function Name: get_frontier
Inputs: num_assets - the number of stocks bounds - 
Returns: list of arrays of optimal weights
Desc: gets the returns and then runs an optimization algorithm in order to
      get minimal volatility associated with each level of returns between
      the max returns and min returns
'''
def gen_frontier(num_assets, bounds):
    returns = calculate_daily_returns(stock_data) # get daily returns
    returns = pow(np.prod(returns), 252/len(returns)) - 1 # Average annualised returns for each stock
    
    initializer = num_assets * [1./num_assets,] #set even initial weight for each asset
    constraints = ({'type':'eq','fun': lambda x: np.sum(x) - 1})
    for i in range(50):
        
        min_ret=optimize.minimize(minimize_ret, #the function we are trying to minimize
                                 initializer,       #set the starting bounds
                                 method = 'SLSQP',  #the method that will be used for optimization
                                 bounds = bounds,   #set the max and min value of each weight
                                 constraints = constraints) #sets the constraints for which the minimize_vol will be minimized
        max_ret=optimize.minimize(maximize_ret, #the function we are trying to minimize
                                 initializer,       #set the starting bounds
                                 method = 'SLSQP',  #the method that will be used for optimization
                                 bounds = bounds,   #set the max and min value of each weight
                                 constraints = constraints) #sets the constraints for which the minimize_vol will be minimized
    min_ret = min_ret['fun']
    max_ret = returns.max() - max_ret['fun']

    target_ret = np.linspace(min_ret, max_ret, 50) #create 50 points between the max and min values
    weights = [] #create storage for weights
    for target in target_ret: #runs for every point in the 50 targets
        #create a function to find when the minimize function at weights is equal to the target
        constraints = ({'type':'eq','fun': lambda x: portfolio_stats(x)['return']-target},
                   {'type':'eq','fun': lambda x: np.sum(x) - 1}) #ensures all weights are equal to 1
        optimal_vol=optimize.minimize(minimize_vol, #the function we are trying to minimize
                                 initializer,       #set the starting bounds
                                 method = 'SLSQP',  #the method that will be used for optimization
                                 bounds = bounds,   #set the max and min value of each weight
                                 constraints = constraints) #sets the constraints for which the minimize_vol will be minimized
        weights.append(optimal_vol['x']) #store the weights which got the optimal volatility 
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
    daily_returns = calculate_daily_returns(stock_data) # get daily returns
    returns = pow(np.prod(daily_returns), 252/len(daily_returns)) - 1 # Average annualised returns for each stock

    return_arr = [] #create storage arrays
    stdevs = []
    for weight in weights: #run for every portfolio
        return_arr.append(np.sum(np.dot(returns, weight))) #store the portfolio returns
        stdevs.append(np.sqrt(np.dot(weight.T, np.dot(daily_returns.cov() * 252, weight)))) #store the portfolio Standard deviation
    return np.array(stdevs), np.array(return_arr)
#end get_optimals

def get_weights(symbols, min_weight, max_weight):
    weight_bounds = list((min_weight, max_weight) for x in range(len(symbols)))

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

    symbols = ["AAPL", "META", "MSFT", "GOOGL", "AMD", "KO", "J", "ALLY", "BMO"] # Add tickers you would like to be in the portfolios
    # Date range
    min_weight = 0.02
    max_weight = 0.5
    
    start = '2014-01-01'
    end = '2024-01-01'

    global stock_data
    stock_data = fetch_stock_data(symbols, start, end)           # get adjusted closes of stock data in an array
    
    weight_bounds = get_weights(symbols, min_weight, max_weight)
    
    opt_weights = gen_frontier(len(symbols), weight_bounds)         #get the optimal weights
    optimal_vol, opt_rets = get_optimals(opt_weights)#get the optimal volatilities and returns
    plot_portfolios(symbols, opt_weights, optimal_vol, opt_rets)            # plot the portfolios on the graph

#end main
    

#run main method
if __name__ == "__main__":
    main()