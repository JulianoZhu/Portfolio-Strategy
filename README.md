# Portfolio-Strategy
I start with the classical Markowitz model of portfolio optimization. I am trading only stocks S1, . . . , Sn, and the total number of stocks is n = 20. There are 12 holding periods and each holding period lasts 2 months (2 years in total). As a result, we can re-balance our portfolio at most 12 times.

Each transaction has a cost composed of a variable portion only. The variable fee is due to the difference between the selling and bidding price of a stock, and is 0.5% of the traded volume. This means that if a stock is quoted at 30 dollars and the investor buys 10 shares, then they are going to pay 0.005·30·10 = 1.50 dollars on top of the 300 dollars the 10 shares would cost otherwise. If they sell 20 shares at the same 30 dollar price then they get 600 dollars, minus the 0.005 · 600 = 3 dollar transaction cost. I need to take them into account when computing portfolio value.

The goal of the optimization effort is to create a tool that allows the user to make regular decisions about re-balancing their portfolio and compare different investment strategies. The user wants to consider the total return, the risk and Sharpe ratio. They may want to minimize/maximize any of these components, while limiting one or more of the other components. The basic building block is a decision made at the first trading day of each 2-month holding period: given a current portfolio, the market prices on that day, and the estimates of the mean and covariance of the daily returns, make a decision about what to buy and sell according to a strategy. I am proposed to test four strategies:

1. “Buy and hold” strategy;
2. “Equally weighted” (also known as “1/n”) portfolio strategy; 
3. “Minimum variance” portfolio strategy;
4. “Maximum Sharpe ratio” portfolio strategy.

The value of the initial portfolio is about one million dollars. The initial cash account is zero USD. The cash account must be nonnegative at all times. The optimization algorithm may suggest buying or selling a non-integer number of shares, which is not allowed. Therefore I rounded the number of shares bought or sold to integer values and subtract transaction fees. If the value of your portfolio after re-balancing plus the transaction fees are smaller than the portfolio value before re-balancing, the remaining funds are accumulated in the cash account. Cash account does not pay any interest, but cash funds should be used toward stock purchases when portfolio is re-balanced next time.
