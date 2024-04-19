import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

def calculate_daily_returns(prices):
    return prices.pct_change().dropna()

def treynor_mazuy_regression(prices_portfolio, prices_market, risk_free_rate):
    returns_portfolio = calculate_daily_returns(prices_portfolio) #* 100
    returns_market = calculate_daily_returns(prices_market) #* 100
    # risk_free_rate *= 100
    
    excess_portfolio_returns = np.array(returns_portfolio.values - risk_free_rate.values)
    excess_market_returns =  np.array(returns_market.values - risk_free_rate.values)

    excess_market_squared = excess_market_returns ** 2

    # Transform features to include quadratic term
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(excess_market_returns.reshape(-1, 1))

    # Add a constant term to the independent variable (market returns)
    X = np.column_stack((np.ones_like(excess_portfolio_returns), X_poly))

    # Fit regression model
    model = LinearRegression().fit(X, excess_portfolio_returns)

    # Extract coefficients
    alpha_p = model.intercept_
    beta_p = model.coef_[1]
    delta_p = model.coef_[2]
    
    return alpha_p, beta_p, delta_p

asset_list = [
    "100 SPY",
    "100 LQD",
    "80-20 SPYxLQD - Daily",
    "60-40 SPYxLQD - Daily",
    "50-50 SPYxLQD - Daily",
    "80-20 SPYxLQD - Monthly",
    "60-40 SPYxLQD - Monthly",
    "50-50 SPYxLQD - Monthly",
    "80-20 SPYxLQD - Yearly",
    "60-40 SPYxLQD - Yearly",
    "50-50 SPYxLQD - Yearly",
    "Fast Algo - Daily",
    "Fast Algo - Monthly",
    "Fast Algo - Yearly",
    "Fast Algo Lintner - Daily",
    "Fast Algo Lintner - Monthly",
    "Fast Algo Lintner - Yearly",
    "Round Robin - Daily"
]

deltas = []
for asset in asset_list:
    PF_Price = pd.read_csv("backtest1.csv", index_col='Date')[asset].iloc[252:]
    MK_Price = pd.read_csv("backtest1.csv", index_col='Date')["100 SPY"].iloc[252:]
    BIL_Ret = pd.read_csv("Asset_Returns.csv", index_col='Date')["BIL"]

    # Example usage:
    alpha, beta, delta = treynor_mazuy_regression(PF_Price, MK_Price, BIL_Ret)
    deltas.append(delta)
    print(asset)
    print("Alpha (αP):", alpha)
    print("Beta (βP):", beta)
    print("Delta (δP):", delta)

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.bar(asset_list, deltas, color='skyblue', edgecolor='black')

ax.set_xlabel('Assets', fontsize=12, fontweight='bold')
ax.set_ylabel('Delta Values', fontsize=12, fontweight='bold')
ax.set_title('Treynor and Mazuy Delta Values for Each Strategy', fontsize=14, fontweight='bold')

ax.tick_params(axis='x', rotation=45, labelsize=10)
ax.tick_params(axis='y', labelsize=10)

ax.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), '{:.2f}'.format(bar.get_height()), 
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()