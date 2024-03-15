#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import timedelta

class Portfolio(object):
	"""docstring for Portfolio"""
	def __init__(self, asset_list = [], index = 'SPY', risk_free = 'BIL', Investments = [],time_stamp0 = 0):
		self.data = pd.read_csv("ETFs_adjclose Feb162024.csv")
		self.asset_list = asset_list
		self.index = index
		self.risk_free = risk_free
		self.returns = self.calc_hist_return()
		self.holdings = {k: v for k, v in zip(asset_list, Investments)}
		self.current_ts = time_stamp0
		self.total_value = self.compute_value()
		self.target_alloc = self.get_asset_alloc()
		self.return_history = []
		self.start = time_stamp0
		self.end = None
        
	def calc_hist_return(self):
		# Calculate daily return of all assets and indices
		dfReturn = self.data.copy()
		dfReturn.loc[:,self.data.columns!='Date'] = self.data.loc[:,self.data.columns!='Date'].pct_change()
		return dfReturn

	def set_start(self,day,month,year):
		for i, row in self.data.iterrows():
			# Extract date from the row and convert it to datetime object
			date = pd.to_datetime(row["Date"], format='%Y/%m/%d')
			# Compare date components (day, month, year)
			# print(date.day,date.month,date.year)
			if date.day == day and date.month == month and date.year == year:
				# print("DATE")
				# print(date)
				self.current_ts = i
				self.total_value = self.compute_value()
				self.target_alloc = self.get_asset_alloc()
				self.return_history = []
				self.start = i
				return

	def set_end(self,day,month,year):
		for i, row in self.data.iterrows():
			# Extract date from the row and convert it to datetime object
			date = pd.to_datetime(row["Date"], format='%Y/%m/%d')
			
			# Compare date components (day, month, year)
			if date.day == day and date.month == month and date.year == year:
				self.end = i
				return

	def set_end_relative(self,years):
		self.end = self.start + years*252

	def is_done(self):
		if self.current_ts >= self.end:
			return True
		return False 

	#Helper function to return total value
	def get_value(self):
		return self.total_value

	#Helper function to return value per holding
	def get_holdings(self):
		return self.holdings

	#Helper function to return percent of PF per holding
	def get_asset_alloc(self):
		normalized = {} 
		for key in self.holdings.keys():
			normalized[key] = self.holdings[key]/self.total_value
		return normalized

	#returns a histogram of returns plotted against the normal curve
	def histogram(self, bins=10):
		# Initialize lists to store annualized returns
		return_history_np = np.array(self.return_history)
		annual_returns = []

		# Pre-calculate the number of possible windows
		num_windows = len(self.return_history) - 252 + 1
		
		# Calculate annualized returns for each one-year moving window
		for i in range(num_windows):
			window_returns = return_history_np[i:i + 252]
			annual_return = np.prod(1 + window_returns) - 1
			annual_returns.append(annual_return)

		frequencies, bins, _ = plt.hist(annual_returns, bins, edgecolor='black')  # Adjust the number of bins as needed
		plt.xlabel('Percentage Return')
		plt.ylabel('Frequency')
		plt.title('Histogram of Percentage Returns')
		plt.grid(True)

		# Plotting the normal distribution curve for all percentage returns
		mu, sigma = np.mean(annual_returns), np.std(annual_returns)
		x = np.linspace(min(annual_returns), max(annual_returns), 100)
		# Scale the normal curve by the maximum frequency of the histogram
		max_freq = max(frequencies)
		plt.plot(x, norm.pdf(x, mu, sigma) * max_freq/norm.pdf(mu, mu, sigma), color='red')

		plt.show()


	#Helper function to compute total value of PF from holdings at current TS
	def compute_value(self):
		value = 0 
		for asset in self.asset_list:
			value += self.holdings[asset]
		return value
	
	#update the value of PF holdings my multiplying value of current holding in that asset by V(i+1)/V(i) = Return for that asset
	def update_next(self):
		current = self.total_value
		for asset in self.asset_list:
			return_i = (self.data[asset][self.current_ts+1])/(self.data[asset][self.current_ts])
			self.holdings[asset] *= return_i
		self.current_ts +=1
		self.total_value = self.compute_value()
		self.return_history.append(self.total_value/current-1)

	def rebalance(self,transaction_cost=0.02):
		target_allocation = self.target_alloc
		for rebalancing_round in range(10):
			current_allocation = self.get_asset_alloc()
			difference = {asset: target_allocation[asset] - current_allocation.get(asset, 0) for asset in set(current_allocation) | set(target_allocation)}
			rebalancing_amount = {}
			for asset in self.asset_list:
				if difference[asset] < 0: 
					rebalancing_amount[asset] = difference[asset] * self.total_value * (1-transaction_cost)**2
				else:
					rebalancing_amount[asset] = difference[asset] * self.total_value
				self.holdings[asset] += rebalancing_amount[asset]
		self.total_value = self.compute_value()
        
	def sharpe_regression(self,window = 252,lamb_obs = 0.995,lamb_sig2_m = 0.94,lamb_sig2_return = 0.94):
		# Use 1-year rolling window for exponentially weighted regression
		dfTraining = self.returns.loc[self.current_ts-window+1:self.current_ts,:]
		# Calculate EWMA market (index) volatility
		weights_sig2_m = np.flip(np.power(lamb_sig2_m, np.arange(window)[::-1]))
		sig2_m = np.inner(dfTraining[self.index]**2,weights_sig2_m)/np.sum(weights_sig2_m)
        
		dfParams = pd.DataFrame(np.nan, index=self.asset_list, columns=['return','alpha','beta','mse'])
        
		weights_obs = np.flip(np.power(lamb_obs, np.arange(window)[::-1]))
		for asset in self.asset_list:
			model = LinearRegression()
			# X is daily market index return, y is daily asset return
			X = dfTraining[self.index].values.reshape(-1,1)
			y = dfTraining[asset].values
			# Weight observations exponentially (recent obs have more weights)
			model = sm.WLS(y, sm.add_constant(X), weights=weights_obs)
			results = model.fit()
			dfParams.loc[asset,'return'] = np.inner(dfTraining[asset].values, weights_obs)/np.sum(weights_obs)
			dfParams.loc[asset,'alpha'] = results.params[0]
			dfParams.loc[asset,'beta'] = results.params[1]
			predictions = results.predict()
			dfParams.loc[asset,'mse'] = np.mean((y - predictions) ** 2)
            
		# Calculate risk-free return
		risk_free_return = np.inner(dfTraining[self.risk_free].values, weights_obs)/np.sum(weights_obs)
            
		return dfParams, risk_free_return, sig2_m
    

	def fast_algo_long(self):
		# Get parameters beta, average return from linear regression
		dfParams, rf, sig2_m = self.sharpe_regression()
		#dfParams['asset_number'] = range(len(dfParams))
		# FAST Algorithm
		dfParams['excess_return'] = dfParams['return']-rf
		dfParams['excess_return_over_beta'] = dfParams['excess_return'].div(dfParams['beta'])
        
		# Sort the dataframe by excess return over beta
		dfSorted = dfParams.sort_values(by='excess_return_over_beta', ascending=False)
		dfSorted['Ci_numerator'] = (dfSorted['excess_return'].mul(dfSorted['beta'])).div(dfSorted['mse'])
		dfSorted['Ci_denominator'] = dfSorted['beta'].pow(2).div(dfSorted['mse'])
		dfSorted['Ci'] = (sig2_m*dfSorted['Ci_numerator'].cumsum()).div(
			1+sig2_m*dfSorted['Ci_denominator'].cumsum())
        
		# Select only assets with excess return over beta higher than Ci    
		dfPortfolio = dfSorted.loc[dfSorted['excess_return_over_beta']>dfSorted['Ci'],:].copy()
		C_cutoff = dfPortfolio.iloc[-1]['Ci']
		dfPortfolio['Zi'] = (dfPortfolio['beta'].div(dfPortfolio['mse'])).mul(
			dfPortfolio['excess_return_over_beta'].sub(C_cutoff))
		dfPortfolio['Xi'] = dfPortfolio['Zi'].div(dfPortfolio['Zi'].sum(axis=0))
        
		# Calculate the optimal allcoation
		fast_alloc = {asset: 0 for asset in self.asset_list}
		for asset in dfPortfolio.index:
			fast_alloc[asset] = dfPortfolio.loc[asset,'Xi']   
		self.target_alloc = fast_alloc

	#helper function that returns true every time our current timestep meets one of our d/w/m/y intervals
	def is_first_of(self,interval):
		date_str = self.data.iloc[self.current_ts]["Date"]
		date = pd.to_datetime(date_str, format='%Y/%m/%d')
		if interval == "year" or interval == "Year" or interval == "y" or interval == "Y":
			if date.day == 1 and date.month == 1:
				return True
			return False
		if interval == "month" or interval == "Month" or interval == "m" or interval == "M":
			if date.day == 1:
				return True
			return False
		if interval == "week" or interval == "Week" or interval == "w" or interval == "W":
			if date.weekday() == 0:
				return True
			return False
		if interval == "day" or interval == "Day" or interval == "d" or interval == "D":
			return True
		print("Key Error: invalid input" + str(interval))
		return None