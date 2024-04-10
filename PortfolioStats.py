import numpy as np
class PortfolioStats:
        
    def __init__(self, daily_ret):
        self.daily_ret = daily_ret
    '''
    Function Name: portfolio_stats
    Inputs: weights
    Returns: port_return - the return associated with the portfolio
            port_vol - the volatility associated with the portfolio
            sharpe - the sharpe rate associated with the portfolio
    Desc: Takes in weights, then calculates and returns the portfolio's
        return, volatility, and sharpe ratio associated with those weights
    '''
    def portfolio_stats(self, weights):
        weights = np.array(weights)                  # set weights into a numpy array, if it is a list
        port_return = np.sum(np.dot((pow(np.prod(self.daily_ret), 252/len(self.daily_ret)) - 1), weights)) #calculate annualized return per stock and multiplys by weights and adds up returns
        port_vol = np.sqrt(np.dot(weights.T, np.dot(self.daily_ret.cov() * 252, weights))) # calculates standard deviation of the portfolio
        sharpe = port_return/port_vol # calculates sharpe ratio
        return {'return': port_return, 'volatility': port_vol, 'sharpe': sharpe}
    #end portfolio_stats