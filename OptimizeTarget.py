import scipy.optimize as optimize
import numpy as np

import MinimizeVolume

class OptimizeTarget:
    def __init__(self, initializer, bounds, lock, portfolioStats, daily_ret):
        self.initializer = initializer
        self.bounds = bounds
        self.lock = lock
        self.portfolioStats = portfolioStats
        self.daily_ret = daily_ret

    def target_vol_thread(self, t, result):
        
        constraints = ({'type':'eq','fun': lambda x: self.portfolioStats.portfolio_stats(x)['return']-t},
                   {'type':'eq','fun': lambda x: np.sum(x) - 1}) #ensures all weights are equal to 1
        optimal_vol=optimize.minimize(MinimizeVolume.MinimizeVolume(self.daily_ret).minimize_vol, #the function we are trying to minimize
                                    self.initializer,       #set the starting bounds
                                    method = 'SLSQP',  #the method that will be used for optimization
                                    bounds = self.bounds,   #set the max and min value of each weight
                                    constraints = constraints) #sets the constraints for which the minimize_vol will be minimized
        with self.lock:  # Acquire the lock before accessing the shared resource
            result.append(optimal_vol)
