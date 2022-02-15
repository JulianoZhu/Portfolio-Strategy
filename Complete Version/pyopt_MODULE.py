import numpy as np
from posdef import nearestPD
from pypfopt import EfficientFrontier, CLA, objective_functions
import cvxpy as cp

def get_weights(mean, covariance, frequency):

    req_return = .07 #annualized
    risk_free_rate = 0.02 #annualized
    max_volatility = 0.8 #annualized
    covariance = nearestPD(covariance)
    mean_ann = mean*frequency #annualized
    covariance_ann = covariance*frequency #annualized

    n=mean.size
    
    case = 3
    
    try:
        
        if case == 1:
            #1. Capital market (tangential) port (maximize sharpe ratio). The stocklist should include SHY ETF.
            ef = EfficientFrontier(mean_ann, covariance_ann, weight_bounds=(0, 1))
            ef.add_objective(objective_functions.L2_reg, gamma=1) #gamma default is 1: We can reduce the strength of the L2 regularisation by reducing gamma
            ef.solver = cp.CPLEX
            w_dict=ef.max_sharpe(risk_free_rate=risk_free_rate) #annualized risk_free 
        elif case == 2:
            #2. Flipped Markowitz port (maximize sharpe ratio while setting the volatility no more than X). Use cash in line 51 of pyopt_MAIN.py to control risk further.
            ef = EfficientFrontier(mean_ann, covariance_ann, weight_bounds=(0, 1))
            ef.solver = cp.CPLEX
            ef.min_volatility() #check viability of maximum volatlity constraint
            o1, sigma, o2 = ef.portfolio_performance(verbose=False);
            ##if the volatility of the minimum variance porfolio is larger than the contraint, get the variance of minimum variance portfolio
            if max_volatility < sigma:
                print("got the minimum variance portfolio")
                ef = EfficientFrontier(mean_ann, covariance_ann, weight_bounds=(0, 1))
                ef.solver = cp.CPLEX
                w_dict=ef.min_volatility()
            else:
                ef = EfficientFrontier(mean_ann, covariance_ann, weight_bounds=(0, 1)) #Note: a new EfficientFrontier object should be instantiated if you want to make any change to objectives/constraints/bounds/parameters.
                ef.add_objective(objective_functions.L2_reg, gamma=1) #gamma default is 1: We can reduce the strength of the L2 regularisation by reducing gamma
                ef.solver = cp.CPLEX
                try:
                    #w_dict=ef.efficient_risk(target_volatility=max_volatility/np.sqrt(frequency)) #non-annualized target_volatility
                    w_dict=ef.efficient_risk(target_volatility=max_volatility) #annualized target_volatility
                except Exception as e:
                    print(e)
                    
        elif case == 3:
            #3. Classic Markowitz port (minimize volatility while setting required return no less than X)
            ef = EfficientFrontier(mean_ann, covariance_ann, weight_bounds=(0, 1))
            ef.add_objective(objective_functions.L2_reg, gamma=.001) #gamma default is 1: We can reduce the strength of the L2 regularisation by reducing gamma
            ef.solver = cp.CPLEX
            if np.max(mean_ann) < req_return: #check viability of minimum required return constraint
                req_return = np.max(mean_ann)  #if the minimum required return is larger than the maximum expected return, replace constraint by maximum expected return
            w_dict=ef.efficient_return(target_return=req_return) #annualized target_return
        elif case == 4:
            #4. Minimum variance port (minimize volatility)
            ef = EfficientFrontier(mean_ann, covariance_ann, weight_bounds=(0, 1))
            ef.solver = cp.CPLEX
            w_dict=ef.min_volatility()
        elif case == 5:
            #5. Critical line algorithm capital market (tangential) port (maximize sharpe ratio)
            co = CLA(mean, covariance, weight_bounds=(0, 1))
            co.solver = cp.CPLEX
            w_dict = co.max_sharpe() #can get stuck in infinite loop, you need to press stop or restart kernel, better run this with our separate program
        else:
            pass
        
        weights = np.array(list(w_dict.values()))
    except:
        weights = np.full(n, 1/n)
        print("error")
    
    return weights



"""

StockList = ['BND','SHY', 'AXP', 'AAPL', 'BA','CAT','CVX','CSCO','KO','DIS','XOM','GS', 'HD', 'IBM','INTC','JNJ','JPM','MCD','MRK','MSFT','NKE','PFE', 'PG', 'TRV','UTX','UNH', 'VZ','V','WMT','WBA']

1. Capital market (tangential) port (maximize sharpe ratio):
TotaAnnReturn in percent = 19.060337
CAGR in percent = 10.929477
Sharpe2 Ratio = 1.190000
Annualized volatility in percent = 9.250000

2.Flipped Markowitz port (maximize sharpe ratio while setting the volatility no more than X):
TotaAnnReturn in percent = 36.949004
CAGR in percent = 16.094207
Sharpe2 Ratio = 1.330000
Annualized volatility in percent = 11.990000
    
3. Classic Markowitz port (minimize volatility while setting required return no less than X)
TotaAnnReturn in percent = 8.517186
CAGR in percent = 6.234225
Sharpe2 Ratio = 2.200000
Annualized volatility in percent = 2.820000

4. Minimum variance port (minimize volatility)
TotaAnnReturn in percent = 4.301825
CAGR in percent = 3.599487
Sharpe2 Ratio = 1.920000
Annualized volatility in percent = 1.880000

5. Critical line algorithm capital market (tangential) port (maximize sharpe ratio)


"""
