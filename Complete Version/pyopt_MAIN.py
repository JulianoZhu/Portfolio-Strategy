import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import pyopt_MODULE

def plot2D(x,y,xLabel='',yLabel='',title='',pathChart=None):
    import matplotlib.pyplot as mpl
    fig=mpl.figure()
    ax=fig.add_subplot(1,1,1) #one row, one column, first plot
    ax.plot(x,y,color='blue')
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel,rotation=90)
    mpl.xticks(rotation='vertical')
    mpl.title(title)
    if pathChart==None:
        mpl.show()
    else:
        mpl.savefig(pathChart)
    mpl.clf() # reset pylab
    return

#################################################################################
## Read stock prices data from csv file
#StockList = ['AAPL','IBM','MSFT','GOOG','QCOM']		# Stock ticker names
#price = pd.read_csv('StockPrices.csv')
#price["Date"]=pd.to_datetime(price["Date"])
#price.index = price["Date"]
#price = price.drop('Date',1)
##################################################################################

pd.core.common.is_list_like = pd.api.types.is_list_like #datareader problem probably fixed in next version of datareader
from pandas_datareader import data as pdr
import datetime as dt
import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)
#start, end = dt.datetime(2000, 1, 3), dt.datetime(2014, 12, 31)
#StockList = ['AAPL','IBM','MSFT','GOOG','QCOM']		# Stock ticker names

#28 Dow Jones Stocks Plus BND ETF (general bonds) and SHY ETF (very short bonds)
start, end = dt.datetime(2009, 1, 30), dt.datetime(2020, 1, 30)
#StockList = ['AXP', 'AAPL', 'BA','CAT','CVX','CSCO','KO','DIS','XOM','GS', 'HD', 'IBM','INTC','JNJ','JPM','MCD','MRK','MSFT','NKE','PFE', 'PG', 'TRV','RTX','UNH', 'VZ','V','WMT','WBA']
#StockList = ['SHY', 'AXP', 'AAPL', 'BA','CAT','CVX','CSCO','KO','DIS','XOM','GS', 'HD', 'IBM','INTC','JNJ','JPM','MCD','MRK','MSFT','NKE','PFE', 'PG', 'TRV','RTX','UNH', 'VZ','V','WMT','WBA']
StockList = ["FDN","IBB","IEZ","IGV","IHE","IHF","IHI","ITA","ITB","IYJ","IYT","IYW","IYZ","KBE","KCE","KIE","PBJ","PBS","SMH","VNQ"]
price = pd.DataFrame()
for ticker in StockList:
    price[ticker]  = pdr.get_data_yahoo(ticker, start=start, end=end).loc[:,'Adj Close'] 
interest_rate = .02  #Fixed interest rate (annual)

cash = 0.0
# Specify number of days to shift
shift = 21 #monthly rebalancing and return period
lookback = 60
T = np.round(252/shift)


interest_rate = interest_rate/T		# Fixed interest rate per return period T

# Compute returns over the time period specified by shift
shift_returns = price/price.shift(shift) - 1

NumStocks = len(StockList)

# Compute mean
shift_returns_mean_ewm = shift_returns.ewm(span=lookback).mean()
shift_returns_mean_sma = shift_returns.rolling(window=lookback).mean()

"""
#covariance matrix construction based on returns
arr = np.zeros(shift_returns.shape[0], dtype=object)
for i in range(0, shift_returns.shape[0]-shift): 
    arr[i+shift]= shift_returns.iloc[i:i+shift].cov().fillna(value=0).to_numpy() 

#covariance matrix construction based on simple moving average of returns
arr = np.zeros(shift_returns_mean_sma.shape[0], dtype=object)
for i in range(0, shift_returns_mean_sma.shape[0]-shift): 
    arr[i+shift]= shift_returns_mean_sma.iloc[i:i+shift].cov().fillna(value=0).to_numpy() 
    
"""
exp_mov_av = False #DO NOT USE EXP MOV EXERCISE FOR HOMEWORK
#covariance matrix construction exponentially weighted moving average returns
arr = np.zeros(shift_returns_mean_ewm.shape[0], dtype=object)
for i in range(0, shift_returns_mean_ewm.shape[0]-shift): 
    if exp_mov_av == True:
        arr[i+shift]= shift_returns_mean_ewm.iloc[i:i+shift].cov().fillna(value=0).to_numpy() 
    else:
        arr[i+shift] = shift_returns.iloc[i:i+shift].cov().fillna(value=0).to_numpy() 
 
shift_returns_copy = shift_returns.copy(deep=True)
shift_returns_copy["covmatrix"] = arr

# Variable Initialization
start_date = '2009-06-01'
index = shift_returns.index
start_index = index.get_loc(start_date)
end_date = index[-1]
end_index = index.get_loc(end_date)
date_index_iter = start_index
StockList.append('InterestRate') ############################ ADD INTEREST RATE
distribution = DataFrame(index=StockList)
returns = Series(index=index)

# Start Value
total_value = 1.0
returns[index[date_index_iter]] = total_value
exp_mov_av = True
while date_index_iter + shift < end_index:
    date = index[date_index_iter] 
    if exp_mov_av == True:
        #mean = shift_returns_mean_ewm.loc[date].to_numpy().reshape(NumStocks,1) 
        mean = shift_returns_mean_sma.loc[date].to_numpy().reshape(NumStocks,1)
    else:
        mean = shift_returns.loc[date].to_numpy().reshape(NumStocks,1) 
    covar = shift_returns_copy["covmatrix"].loc[date]
    w = pyopt_MODULE.get_weights(mean, covar, T)

    w = (1-cash)*w
    w = np.append(w, cash)
    portfolio_alloc = w

    distribution[date.strftime('%Y-%m-%d')] = portfolio_alloc
    
    # Calculating portfolio return
    date2 = index[date_index_iter+shift]
    temp1 = price.loc[date2]/price.loc[date]
    temp1.loc[StockList[-1]] = interest_rate+1 #StockList[-1] is "InterestRate"
    total_value = total_value*np.sum(w*temp1)

	# Increment Date
    date_index_iter += shift
    returns[index[date_index_iter]] = total_value

# Remove dates that there are no trades from returns
returns = returns[np.isfinite(returns)]


# Plot portfolio allocation of last 10 periods
ax = distribution.T.iloc[-10:].plot(kind='bar',stacked=True)
plt.ylim([0,1])
plt.xlabel('Date')
plt.ylabel('distribution')
plt.title('distribution vs. Time')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig('allocation.png')


# Plot portfolio returns vs. time
plt.figure()
returns.plot()
plt.xlabel('Date')
plt.ylabel('Portolio acc. returns')
plt.title('Portfolio acc. returns vs. Time')
# plt.savefig('returns.png')

plt.show()

df = pd.DataFrame(returns, index=returns.index, columns=['I'])
df["returns"]=(df.I-df.I.shift(1))/df.I.shift(1)
start_date = df.iloc[0].name
end_date = df.iloc[-1].name
days_per_month = float(30) #since .days (see below) is calendar days, need to use 30 here
months = float((end_date - start_date).days/days_per_month )
periods = float(12) #12 months in a year
start_val = float(df['I'].iat[0])
end_val = float(df['I'].iat[-1])
years = float(months)/float(periods)

CAGR = ((((end_val/start_val)**(1/years)))-1)
TotaAnnReturn = (end_val-start_val)/(start_val)/years

#try:
#    sharpe =  TotaAnnReturn/( (df.returns.std()) * np.sqrt(periods))
#except ZeroDivisionError:
#    sharpe = 0.0

#this is a more conservative estimate:
try:
    sharpe2 =  (df.returns.mean()/df.returns.std()) * np.sqrt(T)
except ZeroDivisionError:
    sharpe2 = 0.0
    
volat = df.returns.std()*np.sqrt(T)

print ("TotaAnnReturn in percent = %f" %(TotaAnnReturn*100))
print ("CAGR in percent = %f" %(CAGR*100))
#print ("Sharpe Ratio = %f" %(round(sharpe,2)))
print ("Sharpe2 Ratio = %f" %(round(sharpe2,2)))
print ("Annualized volatility in percent = %f" %(round(volat*100,2)))

"""
# Plot stock prices and shifted returns
fig, axes = plt.subplots(nrows=2,ncols=1)
price.plot(ax=axes[0])
shift_returns.plot(ax=axes[1])
axes[0].set_title('Stock Prices')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Price')
axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
axes[1].set_title(str(shift)+ ' Day Shift returns')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('returns ' + str(shift) + ' Days Apart')
axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig('stocks.png', pad_inches=1)
fig.tight_layout()
plt.show()
"""