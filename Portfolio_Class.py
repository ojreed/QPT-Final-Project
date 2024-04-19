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
import seaborn as sns

class Portfolio(object):
	"""docstring for Portfolio"""
	def __init__(self, pf_name = None, asset_list = [], index = 'SPY', risk_free = 'BIL', Investments = [],time_stamp0 = 0):
		#setup
		self.pf_name = pf_name
		self.data = pd.read_csv("ETFs_adjclose Feb162024.csv")
		self.asset_list = asset_list
		#core values
		self.index = index
		self.risk_free = risk_free
		self.returns = self.calc_hist_return()
		self.holdings = {k: v for k, v in zip(asset_list, Investments)}
		self.total_value = self.compute_value()
		self.target_alloc = self.get_asset_alloc()
		self.return_history = []
		self.BM_return_history = []
		self.BM = 'BIL'
		self.BMBool = False
		#time series info
		self.current_ts = time_stamp0
		self.start = time_stamp0
		self.end = None
		self.era = 1
		self.era_track = []
		self.era_price = []
		self.var_threshold = 0

	#generates a smoothed time series for the given past time frame
	def smooth_ts(self,days,display=False):
		# Retrieve the index of the last date
		end_date_index = self.current_ts
		
		# Filter the DataFrame to include only the n previous entries prior to the last date
		filtered_data = self.data.loc[:end_date_index].tail(days)
		
		if display:
			# Melt the DataFrame to long-form
			melted_data = filtered_data.reset_index().melt(id_vars='Date', value_vars=self.asset_list, var_name='Asset', value_name='Value')

			# Set plot size
			sns.set(rc={'figure.figsize': (11.7, 8.27)})

			# Plot the time series
			sns.lineplot(x='Date', y='Value', hue='Asset', data=melted_data)

			# Sparsely label x-axis with dates
			plt.xticks(ticks=np.arange(0, len(filtered_data), step=len(filtered_data)//10), labels=filtered_data['Date'].iloc[np.arange(0, len(filtered_data), step=len(filtered_data)//10)], rotation=45)

			# Show the plot
			plt.show()


		#smooth the data
		for asset in self.asset_list:
			x = np.arange(days)
			y = filtered_data[asset]
			rft = np.fft.rfft(y)
			rft[5:] = 0   # Note, rft.shape = 21
			filtered_data[asset] = np.fft.irfft(rft)
		if display:
			# Melt the DataFrame to long-form
			melted_data = filtered_data.reset_index().melt(id_vars='Date', value_vars=self.asset_list, var_name='Asset', value_name='Value')

			# Set plot size
			sns.set(rc={'figure.figsize': (11.7, 8.27)})

			# Plot the time series
			sns.lineplot(x='Date', y='Value', hue='Asset', data=melted_data)

			# Sparsely label x-axis with dates
			plt.xticks(ticks=np.arange(0, len(filtered_data), step=len(filtered_data)//10), labels=filtered_data['Date'].iloc[np.arange(0, len(filtered_data), step=len(filtered_data)//10)], rotation=45)

			# Show the plot
			plt.show()

		return filtered_data

	# def detect_bull(self,days=40):
	# 	end_date_index = self.current_ts
		
	# 	# Filter the DataFrame to include only the n previous entries prior to the last date
	# 	filtered_data = self.data["SPY"].loc[:end_date_index].tail(150)
	# 	#smooth the data
	# 	x = np.arange(days)
	# 	y = filtered_data
	# 	rft = np.fft.rfft(y)
	# 	rft[5:] = 0   # Note, rft.shape = 21
	# 	filtered_data = np.fft.irfft(rft)

	# 	# Calculate returns
	# 	returns = [(filtered_data[i] - filtered_data[i-1]) / filtered_data[i-1] for i in range(1, 150)]
		
	# 	if len(returns) < days:
	# 		return None  # Not enough data to calculate average return for n days
		
	# 	# Calculate average return for the past n days
	# 	avg_return_n_days = sum(returns[-days:]) / days
		
	# 	# Calculate average return for the full m days
	# 	avg_return_m_days = sum(returns) / len(returns)
		
	# 	# Check if average return for the past n days is above average return for the full m days
	# 	if avg_return_n_days > avg_return_m_days:
	# 		return 1
	# 	else:
	# 		return 0

	# def calibrate_bull(self,days=40,quantile=0.95):
	# 	df = pd.DataFrame()
	# 	df['Price'] = self.data["SPY"].loc[:self.current_ts]
	# 	# Calculate returns
	# 	df['Return'] = df['Price'].pct_change()

	# 	# Calculate n day average return
	# 	n = days
	# 	df['N_day_average_return'] = df['Return'].rolling(window=n).mean()

	# 	# Calculate variance up to that day
	# 	df['Variance_up_to_day'] = df['Return'].expanding().var()

	# 	# Calculate n day variance
	# 	df['N_day_variance'] = df['Return'].rolling(window=n).var()

	# 	self.var_threshold = df['N_day_variance'].quantile(quantile)

		

	# def detect_bull(self,days=40):
	# 	df = pd.DataFrame()
	# 	df['Price'] = self.data["SPY"].loc[:self.current_ts].tail(days)
	# 	df['Return'] = df['Price'].pct_change()
	# 	if df['Return'].var() >= self.var_threshold:
	# 		return 0
	# 	return 1

	def calibrate_bull(self, days=40, quantile=0.95):
		df = pd.DataFrame()
		df['Price'] = self.data["SPY"].loc[:self.current_ts]
		# Calculate returns
		df['Return'] = df['Price'].pct_change()

		# Calculate n day average return
		n = days
		df['N_day_average_return'] = df['Return'].rolling(window=n).mean()

		# Calculate semivariance up to that day
		df['Semivariance_up_to_day'] = df[df['Return'] < 0]['Return'].expanding().var()

		# Calculate n day semivariance
		df['N_day_semivariance'] = df[df['Return'] < 0]['Return'].rolling(window=n).var()

		self.semivar_threshold = df['N_day_semivariance'].quantile(quantile)

	def detect_bull(self, days=40):
		df = pd.DataFrame()
		df['Price'] = self.data["SPY"].loc[:self.current_ts].tail(days)
		df['Return'] = df['Price'].pct_change()
		if df[df['Return'] < 0]['Return'].var() >= self.semivar_threshold:
			return 0
		return 1

	def Round_Robin(self,days=40):
		if self.detect_bull(days):
			if self.era != 1:
				self.era = 1
				# print("bull")
				self.asset_list = ['XLE','XLF','XLK','XLV','XLI','XLY','XLP','XLU','XLB']
				self.holdings = {k: v for k, v in zip(self.asset_list, [self.total_value/len(self.asset_list)]*len(self.asset_list))}
				self.target_alloc = self.get_asset_alloc()
				# print(self.holdings)
				self.rebalance(0.004)
			
			self.momentum_rebalance(252,15,3)
		else:
			if self.era == 1:
				self.era = 0
				# print("bear")
				self.asset_list = ["LQD","BIL","SHY","IEF","TLT","VGSH","VGIT","VGLT","BND"]
				self.holdings = {k: v for k, v in zip(self.asset_list, [self.total_value/len(self.asset_list)]*len(self.asset_list))}
				self.target_alloc = self.get_asset_alloc()
				# print(self.holdings)
				self.rebalance(0.004)
			self.momentum_rebalance(252,15,4)
		self.era_track.append(self.era)
		self.era_price.append(self.data["SPY"].iloc[self.current_ts])

	def momentum_rebalance(self, smoothing_period=252 ,lookback_period=7,keep=1):
		returns = self.smooth_ts(smoothing_period,False)
		momentum = returns[self.asset_list].pct_change(lookback_period).iloc[-1]  # Calculate returns over lookback period
		momentum = momentum.sort_values(ascending=False)
		i = 0
		for asset_name, momentum_value in momentum.items():
			if i < keep:
				self.target_alloc[asset_name] = 1/keep
			else: 
				self.target_alloc[asset_name] = 0
			i += 1
		self.rebalance(0.004)
		
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
			# print(i)
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

	def get_date(self):
		return self.data['Date'].iloc[self.current_ts]


	#returns a histogram of returns plotted against the normal curve
	def histogram(self, bins=15):
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

		if (self.BMBool):

			# Initialize lists to store annualized returns
			BMreturn_history_np = np.array(self.BM_return_history)
			BMannual_returns = []

			# Pre-calculate the number of possible windows
			num_windows = len(self.return_history) - 252 + 1
			
			# Calculate annualized returns for each one-year moving window
			for i in range(num_windows):
				BMwindow_returns = BMreturn_history_np[i:i + 252]
				BMannual_return = np.prod(1 + BMwindow_returns) - 1
				BMannual_returns.append(BMannual_return)
			print(annual_returns)
			for i in range(len(annual_returns)):
				annual_returns[i] -= BMannual_returns[i]


		frequencies, bins, _ = plt.hist(annual_returns, bins, edgecolor='black')  # Adjust the number of bins as needed
		plt.xlabel('Percentage Return')
		plt.ylabel('Frequency')
		plt.title('Histogram of Percentage Returns - ' + str(self.pf_name))
		plt.grid(True)

		# Plotting the normal distribution curve for all percentage returns
		mu, sigma = np.mean(annual_returns), np.std(annual_returns)
		x = np.linspace(min(annual_returns), max(annual_returns), 100)
		# Scale the normal curve by the maximum frequency of the histogram
		max_freq = max(frequencies)
		plt.plot(x, norm.pdf(x, mu, sigma) * max_freq/norm.pdf(mu, mu, sigma), color='red')

		plt.show()

		# # Set the file path where you want to save the plot
		# file_path = str(self.pf_name) + '_histogram.png'  # You can change the extension to any supported image format (e.g., .png, .jpg, .pdf)

		# # Save the plot directly to the file without displaying it
		# plt.savefig(file_path, dpi=300, bbox_inches='tight')

		# # Close the plot to free up memory (optional)
		# plt.close()

	def plot_ret(self):
		fig, ax = plt.subplots(figsize=(10, 6))
		for i in range(len(self.era_price) - 1):
			color = 'red' if self.era_track[i] == 0 else 'green'
			ax.plot([i, i+1], [self.era_price[i], self.era_price[i+1]], color=color)

		ax.set_xlabel('Time')
		ax.set_ylabel('SPY Price')
		ax.set_title('Bear vs Bull Markets on SPY')
		ax.grid(True)
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
		self.BM_return_history.append(self.data[self.BM].iloc[self.current_ts]/self.data[self.BM].iloc[self.current_ts-1]-1)

	def rebalance(self,transaction_cost=0.004):
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
		
	def sharpe_regression(self,window = 252,lamb_obs = 0.99,lamb_sig2_m = 0.94,lamb_sig2_return = 0.94):
		# Use 1-year rolling window for exponentially weighted regression
		dfTraining = self.returns.loc[self.current_ts-window+1:self.current_ts,:]
		# Calculate EWMA market (index) volatility
		weights_sig2_m = Portfolio.weights(lamb_sig2_m, window)
		sig2_m = np.inner(dfTraining[self.index]**2,weights_sig2_m)
		
		dfParams = pd.DataFrame(np.nan, index=self.asset_list, columns=['return','alpha','beta','mse'])
		
		weights_obs = Portfolio.weights(lamb_obs, window)
		for asset in self.asset_list:
			model = LinearRegression()
			# X is daily market index return, y is daily asset return
			X = dfTraining[self.index].values.reshape(-1,1)
			y = dfTraining[asset].values
			# Weight observations exponentially (recent obs have more weights)
			model = sm.WLS(y, sm.add_constant(X), weights=weights_obs)
			results = model.fit()
			dfParams.loc[asset,'return'] = np.inner(dfTraining[asset].values, weights_obs)
			dfParams.loc[asset,'alpha'] = results.params[0]
			dfParams.loc[asset,'beta'] = results.params[1]
			predictions = results.predict()
			dfParams.loc[asset,'mse'] = np.inner((y - predictions) ** 2,weights_obs)
			
		# Calculate risk-free return
		risk_free_return = np.inner(dfTraining[self.risk_free].values, weights_obs)
			
		return dfParams, risk_free_return, sig2_m

	def fast_algo(self, short = False):
		# Get parameters beta, average return from linear regression
		dfParams, rf, sig2_m = self.sharpe_regression()
		#print(dfParams.loc[:,'return'], rf)
		dfParams['excess_return'] = dfParams['return']-rf
		dfParams['excess_return_over_beta'] = dfParams['excess_return'].div(dfParams['beta'])
		
		# Sort the dataframe by excess return over beta
		dfSorted = dfParams.sort_values(by='excess_return_over_beta', ascending=False)
		dfSorted['Ci_numerator'] = (dfSorted['excess_return'].mul(dfSorted['beta'])).div(dfSorted['mse'])
		dfSorted['Ci_denominator'] = dfSorted['beta'].pow(2).div(dfSorted['mse'])
		dfSorted['Ci'] = (sig2_m*dfSorted['Ci_numerator'].cumsum()).div(
			1+sig2_m*dfSorted['Ci_denominator'].cumsum())     
		
		if short == False:
			# Select only assets with excess return over beta higher than Ci    
			dfPortfolio = dfSorted.loc[dfSorted['excess_return_over_beta']>dfSorted['Ci'],:].copy()
			if dfPortfolio.empty: return self.target_alloc
			C_cutoff = dfPortfolio.iloc[-1]['Ci']
			dfPortfolio['Zi'] = (dfPortfolio['beta'].div(dfPortfolio['mse'])).mul(
				dfPortfolio['excess_return_over_beta'].sub(C_cutoff))
			dfPortfolio['Xi'] = dfPortfolio['Zi'].div(dfPortfolio['Zi'].sum(axis=0))

		elif short=='Lintner':
			# Select all assets
			dfPortfolio = dfSorted.copy()
			C_cutoff = dfPortfolio.iloc[-1]['Ci']
			dfPortfolio['Zi'] = (dfPortfolio['beta'].div(dfPortfolio['mse'])).mul(
				dfPortfolio['excess_return_over_beta'].sub(C_cutoff))
			dfPortfolio['Xi'] = dfPortfolio['Zi'].div((dfPortfolio['Zi'].abs()).sum(axis=0))
			# print(dfPortfolio['Xi'])
		
		elif short=='Leveraged':
			# Select all assets
			dfPortfolio = dfSorted.copy()
			C_cutoff = dfPortfolio.iloc[-1]['Ci']
			dfPortfolio['Zi'] = (dfPortfolio['beta'].div(dfPortfolio['mse'])).mul(
				dfPortfolio['excess_return_over_beta'].sub(C_cutoff))
			dfPortfolio['Xi'] = dfPortfolio['Zi'].div(dfPortfolio['Zi'].sum(axis=0))
			
		# Output the optimal allocation
		fast_alloc = {asset: 0 for asset in self.asset_list}
		for asset in dfPortfolio.index:
			fast_alloc[asset] = dfPortfolio.loc[asset,'Xi']
		return fast_alloc


	def fast_algo_long(self):
		# Get parameters beta, average return from linear regression
		dfParams, rf, sig2_m = self.sharpe_regression()
		#print(dfParams.loc[:,'return'], rf)
		dfParams['excess_return'] = dfParams['return']-rf
		dfParams['excess_return_over_beta'] = dfParams['excess_return'].div(dfParams['beta'])
		
		# Sort the dataframe by excess return over beta
		dfSorted = dfParams.sort_values(by='excess_return_over_beta', ascending=False)
		dfSorted['Ci_numerator'] = (dfSorted['excess_return'].mul(dfSorted['beta'])).div(dfSorted['mse'])
		dfSorted['Ci_denominator'] = dfSorted['beta'].pow(2).div(dfSorted['mse'])
		dfSorted['Ci'] = (sig2_m*dfSorted['Ci_numerator'].cumsum()).div(
			1+sig2_m*dfSorted['Ci_denominator'].cumsum())
		#print(dfSorted)       
		
		# Select only assets with excess return over beta higher than Ci    
		dfPortfolio = dfSorted.loc[dfSorted['excess_return_over_beta']>dfSorted['Ci'],:].copy()
		if dfPortfolio.empty: return self.target_alloc
		C_cutoff = dfPortfolio.iloc[-1]['Ci']
		dfPortfolio['Zi'] = (dfPortfolio['beta'].div(dfPortfolio['mse'])).mul(
			dfPortfolio['excess_return_over_beta'].sub(C_cutoff))
		dfPortfolio['Xi'] = dfPortfolio['Zi'].div(dfPortfolio['Zi'].sum(axis=0))
		
		# Calculate the optimal allcoation
		fast_alloc = {asset: 0 for asset in self.asset_list}
		for asset in dfPortfolio.index:
			fast_alloc[asset] = dfPortfolio.loc[asset,'Xi'] 
		self.target_alloc = fast_alloc  
		return fast_alloc
		#print(self.target_alloc)

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
	@staticmethod
	def weights(lamb, window):
		weights_raw = np.power(lamb, np.arange(window)[::-1])
		weights_norm = weights_raw/np.sum(weights_raw)
		return weights_norm