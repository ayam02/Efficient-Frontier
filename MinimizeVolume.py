import PortfolioStats
class MinimizeVolume:
    def __init__(self, daily_ret):
        self.daily_ret = daily_ret
    '''
    Function Name: minimize_vol
    Inputs: weights - array of weights
    Returns: the volatility or std of the portfolio at those weights
    Desc: takes in weights, run portfolio_stats funtion with those weights
        and returns the volitility associated with that portfolio
    '''
    def minimize_vol(self, weights):
        vol = PortfolioStats.PortfolioStats(self.daily_ret).portfolio_stats(weights)['volatility']
        return vol
    #end minimize_vol