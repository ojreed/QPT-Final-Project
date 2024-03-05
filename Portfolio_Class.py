import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from datetime import timedelta


class Portfolio(object):
	"""docstring for Portfolio"""
	def __init__(self, asset_list = [], Investments = [],time_stamp0 = 0):
		self.data = pd.read_csv("ETFs_adjclose Feb162024.csv")
		self.asset_list = asset_list
		self.holdings = {k: v for k, v in zip(asset_list, Investments)}
		self.current_ts = time_stamp0
		self.total_value = self.compute_value()
		self.target_alloc = self.get_asset_alloc()
		self.return_history = []
		self.start = time_stamp0
		self.end = None

	def set_start(self,day,month,year):
		for i, row in self.data.iterrows():
			# Extract date from the row and convert it to datetime object
			date = pd.to_datetime(row["Date"], format='%Y/%m/%d')
			
			# Compare date components (day, month, year)
			if date.day == day and date.month == month and date.year == year:
				self.current_ts = i
				self.total_value = self.compute_value()
				self.target_alloc = self.get_asset_alloc()
				self.return_history = []
				self.start = i

	def set_end(self,day,month,year):
		for i, row in self.data.iterrows():
			# Extract date from the row and convert it to datetime object
			date = pd.to_datetime(row["Date"], format='%Y/%m/%d')
			
			# Compare date components (day, month, year)
			if date.day == day and date.month == month and date.year == year:
				self.end = i

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
		df = pd.DataFrame(self.return_history, columns=['Daily Returns'])
	
		# Calculate the rolling window 1-year (252 trading days) annual returns
		annual_returns = df['Daily Returns'].rolling(window=252).sum()
		frequencies, bins, _ = plt.hist(annual_returns, bins, edgecolor='black')  # Adjust the number of bins as needed
		plt.xlabel('Percentage Return')
		plt.ylabel('Frequency')
		plt.title('Histogram of Percentage Returns')
		plt.grid(True)

		# Plotting the normal distribution curve for all percentage returns
		mu, sigma = np.mean(annual_returns), np.std(annual_returns)
		x = np.linspace(annual_returns.min(skipna=True), annual_returns.max(skipna=True), 100)
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
		current_allocation = self.get_asset_alloc()
		difference = {asset: target_allocation[asset] - current_allocation.get(asset, 0) for asset in set(current_allocation) | set(target_allocation)}
		# Calculate the rebalancing amount for each asset
		rebalancing_amount = {asset: difference[asset] * self.total_value * (1 - transaction_cost) for asset in difference}
		for asset in self.asset_list:
			self.holdings[asset] += rebalancing_amount[asset]
		self.total_value = self.compute_value()

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
