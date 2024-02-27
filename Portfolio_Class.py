import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


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
	def histogram(self,bins=10):
		plt.hist(self.return_history,bins, edgecolor='black')  # Adjust the number of bins as needed
		plt.xlabel('Percentage Return')
		plt.ylabel('Frequency')
		plt.title('Histogram of Percentage Returns')
		plt.grid(True)

		# Plotting the normal distribution curve for all percentage returns
		mu, sigma = np.mean(self.return_history), np.std(self.return_history)
		x = np.linspace(min(self.return_history), max(self.return_history), 100)
		plt.plot(x, norm.pdf(x, mu, sigma), color='red', label='Normal Distribution')


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

