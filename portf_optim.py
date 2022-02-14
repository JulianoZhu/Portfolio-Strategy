# Import libraries
import pandas as pd
import numpy as np
import math
import cplex

# Complete the following functions
def strat_buy_and_hold(x_init, cash_init, mu, Q, cur_prices):
   x_optimal = x_init
   cash_optimal = cash_init
   return x_optimal, cash_optimal

def strat_equally_weighted(x_init, cash_init, mu, Q, cur_prices):
    w = np.ones(20)/20
    Total_v = np.dot(cur_prices, x_init) + cash_init
    x_optimal = np.floor(w * (Total_v/cur_prices))
    trans_cost = np.dot(0.005 * cur_prices, abs(x_optimal - x_init))
    cash_optimal = Total_v - np.dot(cur_prices, x_optimal) - trans_cost
    return x_optimal, cash_optimal

def strat_min_variance(x_init, cash_init, mu, Q, cur_prices):
    Total_v = np.dot(cur_prices, x_init) + cash_init
    
    #set variavle constraints
    c  = [0.0] * 20
    lb = [0.0] * 20
    ub = [1.0] * 20
    
    A = []
    for k in range(20):
        A.append([[0,1],[1.0,0]])
    
    #build cplex optimization
    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    var_names = ["w_%s" % i for i in range(1,20+1)]
    cpx.linear_constraints.add(rhs=[1.0, 0], senses="EG")
    
    cpx.variables.add(obj=c, lb=lb, ub=ub, columns=A, names=var_names)
    Qmat = [[list(range(20)), list(2*Q[k,:])] for k in range(20)]
    
    cpx.objective.set_quadratic(Qmat)
    cpx.parameters.threads.set(6)
    cpx.set_results_stream(None)
    cpx.set_warning_stream(None)
    cpx.solve()
    
    w_optimal = cpx.solution.get_values()
    x_optimal = np.floor(w_optimal * (Total_v/cur_prices))
    trans_cost = np.dot(0.005 * cur_prices, abs(x_optimal - x_init))
    cash_optimal = Total_v - np.dot(cur_prices, x_optimal) - trans_cost
    return x_optimal, cash_optimal

def strat_max_Sharpe(x_init, cash_init, mu, Q, cur_prices):
    Total_v = np.dot(cur_prices, x_init) + cash_init
    #daily risk free rate
    r_rf_daily = (1+r_rf)**(1.0/252)-1 

    Q = np.append(Q,np.zeros((20,1)),axis=1)
    Q = np.vstack([Q,np.zeros((21))])

    diff = mu-r_rf_daily

    A = []
    for k in range(20):
        A.append([[0,1],[diff[k],1.0]])
    A.append([[0,1],[0,-1.0]])

    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    c = [0.0]*21
    lb = [0.0]*21
    ub = [np.inf]*21

    var_names = ['y_%s'% i for i in range(1,21+1)]
    cpx.linear_constraints.add(rhs=[1.0,0],senses='EE')
    cpx.variables.add(obj=c,lb=lb,ub=ub,columns=A,names=var_names)

    Qmat = [[list(range(21)),list(2*Q[k,:])] for k in range(21)]

    cpx.objective.set_quadratic(Qmat)
    cpx.parameters.threads.set(6)
    cpx.set_results_stream(None)
    cpx.set_warning_stream(None)
    cpx.solve()

    y = np.array(cpx.solution.get_values())
    w_optimal = y[0:20]/y[:20].sum()
    
    x_optimal = np.floor(w_optimal * (Total_v/cur_prices))
    trans_cost = np.dot(0.005 * cur_prices, abs(x_optimal - x_init))
    cash_optimal = Total_v - np.dot(cur_prices, x_optimal) - trans_cost

    return x_optimal, cash_optimal
def strat_equally_weighted_BH(x_init, cash_init, mu, Q, cur_prices):
    x_optimal = np.array([ 672., 1011.,   25., 1454., 1034.,  656., 1089., 5237.,   
                          35.,1376., 2518.,  424.,  859.,  377., 1006.,  316.,  
                          830., 720., 1492.,  898.])
    
    cash_optimal = cash_init
    return x_optimal, cash_optimal

def strat_min_variance_BH(x_init, cash_init, mu, Q, cur_prices):    
    x_optimal = np.array([ 614.,    0.,    6.,    0., 2183.,    0.,    0.,    0.,    0.,
                           33.,  610.,  961.,    0.,    0.,    0., 1365.,    0., 2901.,
                           578., 4859.])
    cash_optimal = cash_init
    return x_optimal, cash_optimal

def strat_max_Sharpe_BH(x_init, cash_init, mu, Q, cur_prices):
    x_optimal = np.array([2693., 1123.,    0., 1946.,    0.,    0.,    0.,    0.,    0.,
                       0., 7816.,    0.,    0.,    0.,    0., 1039.,    0., 5107.,
                       0.,    0.]) 
    cash_optimal = cash_init

    return x_optimal, cash_optimal
# Input file
input_file_prices = 'Daily_closing_prices.csv'

# Read data into a dataframe
df = pd.read_csv(input_file_prices)

# Convert dates into array [year month day]
def convert_date_to_array(datestr):
    temp = [int(x) for x in datestr.split('/')]
    return [temp[-1], temp[0], temp[1]]

dates_array = np.array(list(df['Date'].apply(convert_date_to_array)))
data_prices = df.iloc[:, 1:].to_numpy()
dates = np.array(df['Date'])
# Find the number of trading days in Nov-Dec 2019 and
# compute expected return and covariance matrix for period 1
day_ind_start0 = 0
day_ind_end0 = len(np.where(dates_array[:,0]==2019)[0])
cur_returns0 = data_prices[day_ind_start0+1:day_ind_end0,:] / data_prices[day_ind_start0:day_ind_end0-1,:] - 1
mu = np.mean(cur_returns0, axis = 0)
Q = np.cov(cur_returns0.T)

# Remove datapoints for year 2019
data_prices = data_prices[day_ind_end0:,:]
dates_array = dates_array[day_ind_end0:,:]
dates = dates[day_ind_end0:]

# Initial positions in the portfolio
init_positions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 902, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17500])

# Initial value of the portfolio
init_value = np.dot(data_prices[0,:], init_positions)
print('\nInitial portfolio value = $ {}\n'.format(round(init_value, 2)))

# Initial portfolio weights
w_init = (data_prices[0,:] * init_positions) / init_value

# Number of periods, assets, trading days
N_periods = 6*len(np.unique(dates_array[:,0])) # 6 periods per year
N = len(df.columns)-1
N_days = len(dates)

# Annual risk-free rate for years 2020-2021 is 2.5%
r_rf = 0.025

# Number of strategies
strategy_functions = ['strat_buy_and_hold', 'strat_equally_weighted', 'strat_min_variance', 'strat_max_Sharpe', 
                      'strat_equally_weighted_BH', 'strat_min_variance_BH', 'strat_max_Sharpe_BH']
strategy_names     = ['Buy and Hold', 'Equally Weighted Portfolio', 'Mininum Variance Portfolio', 'Maximum Sharpe Ratio Portfolio',
                      'Equally Weighted Portfolio B&H', 'Mininum Variance Portfolio B&H', 
                      'Maximum Sharpe Ratio Portfolio B&H']
#N_strat = 1  # comment this in your code
N_strat = len(strategy_functions)  # uncomment this in your code
fh_array = [strat_buy_and_hold, strat_equally_weighted, strat_min_variance, strat_max_Sharpe, 
            strat_equally_weighted_BH, strat_min_variance_BH, strat_max_Sharpe_BH]

portf_value = [0] * N_strat
x = np.zeros((N_strat, N_periods),  dtype=np.ndarray)
cash = np.zeros((N_strat, N_periods),  dtype=np.ndarray)
#creat list for two charts of strategy 3 and 4 dynamic changes 
w3_allocation = []
w4_allocation = []

for period in range(1, N_periods+1):
   # Compute current year and month, first and last day of the period
   if dates_array[0, 0] == 20:
       cur_year  = 20 + math.floor(period/7)
   else:
       cur_year  = 2020 + math.floor(period/7)

   cur_month = 2*((period-1)%6) + 1
   day_ind_start = min([i for i, val in enumerate((dates_array[:,0] == cur_year) & (dates_array[:,1] == cur_month)) if val])
   day_ind_end = max([i for i, val in enumerate((dates_array[:,0] == cur_year) & (dates_array[:,1] == cur_month+1)) if val])
   print('\nPeriod {0}: start date {1}, end date {2}'.format(period, dates[day_ind_start], dates[day_ind_end]))
   
   # Prices for the current day
   cur_prices = data_prices[day_ind_start,:]

   # Execute portfolio selection strategies
   for strategy in range(N_strat):

      # Get current portfolio positions
      if period == 1:
         curr_positions = init_positions
         curr_cash = 0
         portf_value[strategy] = np.zeros((N_days, 1))
      else:
         curr_positions = x[strategy, period-2]
         curr_cash = cash[strategy, period-2]

      # Compute strategy
      x[strategy, period-1], cash[strategy, period-1] = fh_array[strategy](curr_positions, curr_cash, mu, Q, cur_prices)
     
      ###################### 
      # Verify that strategy is feasible (you have enough budget to re-balance portfolio)
      # Check that cash account is >= 0
      # Check that we can buy new portfolio subject to transaction costs

      ###################### Insert your code here ############################
      while cash[strategy, period-1] < 0:
           
           Total_v = np.dot(cur_prices, curr_positions) + curr_cash

           ratio = x[strategy][period-1]/np.sum(x[strategy][period-1])
           cash_adjust = abs(cash[strategy][period-1]) * ratio
           x_adjust = np.ceil(cash_adjust/cur_prices)
           x[strategy][period-1] = x[strategy][period-1] - x_adjust
           trans_cost = 0.005*np.dot(cur_prices, abs(x[strategy][period-1]-curr_positions))
           cash[strategy][period-1] = Total_v - np.dot(cur_prices,x[strategy][period-1]) - trans_cost

      # Compute portfolio value
      p_values = np.dot(data_prices[day_ind_start:day_ind_end+1,:], x[strategy, period-1]) + cash[strategy, period-1]
      portf_value[strategy][day_ind_start:day_ind_end+1] = np.reshape(p_values, (p_values.size,1))
      print('  Strategy "{0}", value begin = $ {1:.2f}, value end = $ {2:.2f}'.format( strategy_names[strategy], 
             portf_value[strategy][day_ind_start][0], portf_value[strategy][day_ind_end][0]))
      #Append strategy 3 and 4 weight into list
      if strategy == 2:
          w3_allocation.append((cur_prices*x[strategy][period-1])/np.dot(cur_prices,x[strategy][period-1]))
      elif strategy== 3:
          w4_allocation.append((cur_prices*x[strategy][period-1])/np.dot(cur_prices,x[strategy][period-1]))

   # Compute expected returns and covariances for the next period
   cur_returns = data_prices[day_ind_start+1:day_ind_end+1,:] / data_prices[day_ind_start:day_ind_end,:] - 1
   mu = np.mean(cur_returns, axis = 0)
   Q = np.cov(cur_returns.T)

# Plot results
###################### Insert your code here ############################
import matplotlib.pyplot as plt
# daily value of your portfolio
plt.figure(figsize=(20,13))
plt.plot(portf_value[0], label='Buy and hold', color='yellow')
plt.plot(portf_value[1], label='Equally weighted', color='green')
plt.plot(portf_value[2], label='Minimum variance', color='blue')
plt.plot(portf_value[3], label='Maximum Sharpe ratio', color='red')

plt.legend()
plt.title('Daily Values of Portfolio')
plt.xlabel('Day')
plt.ylabel('Daily value of portfolio ($)')
plt.show()

# Dynamic change in allocation
stock_list = df.columns[1:]

plt.figure(figsize=(20,10))
plt.plot(w3_allocation, label=stock_list)
plt.legend()
plt.title('Minimum Variance Portfolio weight Dynamic Changes ')
plt.xlabel('Period')
plt.ylabel('Weight')
plt.show()

plt.figure(figsize=(20,10))
plt.plot(w4_allocation, label=stock_list)
plt.legend()
plt.title('Maximum Sharpe Ratio weight Dynamic Changes ')
plt.xlabel('Period')
plt.ylabel('Weight')
plt.show()

# plot with variation
plt.figure(figsize=(20,13))

plt.plot(portf_value[0], label='Buy and hold', color='yellow')
plt.plot(portf_value[1], label='Equally weighted', color='green')
plt.plot(portf_value[2], label='Minimum variance', color='blue')
plt.plot(portf_value[3], label='Maximum Sharpe ratio', color='red')
plt.plot(portf_value[4], label='Equally weighted B&H', color='black')
plt.plot(portf_value[5], label='Minimum variance B&H', color='purple')
plt.plot(portf_value[6], label='Maximum Sharpe ratio B&H', color='brown')

plt.legend()
plt.title('Daily Values of Portfolio')
plt.xlabel('Day')
plt.ylabel('Daily value of portfolio ($)')
plt.show()
