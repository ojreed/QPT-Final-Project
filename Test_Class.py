import Portfolio_Class as PF
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Test_Class(object):
	"""docstring for Test_Class"""
	def __init__(self, Start_Date = (8,8,2013), Years = 5):
		self.Start_Date = Start_Date
		self.Years = Years
		self.results_df = pd.DataFrame(columns=['Date'])
	
	def test_fixed_alloc(self,test_name,asset_list,Investments,freq = 1):
		print(test_name)
		if freq == 0:
			freq = "day"
		if freq == 1:
			freq = "month"
		if freq == 2:
			freq = "year"	
		test = PF.Portfolio(pf_name=test_name,asset_list=asset_list,Investments=Investments)
		print("Inital Value: " + str(test.get_value()))
		print("Inital Alloc: " + str(test.get_asset_alloc()))
		test.set_start(self.Start_Date[0],self.Start_Date[1],self.Start_Date[2])
		# print(test.current_ts)
		test.set_end_relative(self.Years)
		dates = []
		value = []
		while not test.is_done():
			value.append(test.get_value())
			dates.append(test.get_date())
			if test.is_first_of(freq):
				test.rebalance(0.02)
			test.update_next()
		print("Final Value: " + str(test.get_value()))
		print("Final Alloc: " + str(test.get_asset_alloc()))
		test.histogram(100)
		self.results_df["Date"] = dates
		self.results_df[test_name] = value

	def test_fast_algo(self,test_name,asset_list,Investments,freq = 1):
		print(test_name)
		if freq == 0:
			freq = "day"
		if freq == 1:
			freq = "month"
		if freq == 2:
			freq = "year"	
		test = PF.Portfolio(pf_name=test_name,asset_list=asset_list,Investments=Investments)
		print("Inital Value: " + str(test.get_value()))
		print("Inital Alloc: " + str(test.get_asset_alloc()))
		test.set_start(self.Start_Date[0],self.Start_Date[1],self.Start_Date[2])
		test.set_end_relative(self.Years)
		dates = []
		value = []
		while not test.is_done():
			value.append(test.get_value())
			dates.append(test.get_date())
			if test.is_first_of(freq):
				test.fast_algo_long()
				test.rebalance(0.02)
			test.update_next()
		print("Final Value: " + str(test.get_value()))
		print("Final Alloc: " + str(test.get_asset_alloc()))
		test.histogram(60)
		self.results_df["Date"] = dates
		self.results_df[test_name] = value

	def test_RR(self,test_name,asset_list,Investments,freq = 1):
		print(test_name)
		if freq == 0:
			freq = "day"
		if freq == 1:
			freq = "month"
		if freq == 2:
			freq = "year"	
		test = PF.Portfolio(pf_name=test_name,asset_list=asset_list,Investments=Investments)
		print("Inital Value: " + str(test.get_value()))
		print("Inital Alloc: " + str(test.get_asset_alloc()))
		test.set_start(self.Start_Date[0],self.Start_Date[1],self.Start_Date[2])
		test.set_end_relative(self.Years)
		dates = []
		value = []
		while not test.is_done():
			value.append(test.get_value())
			dates.append(test.get_date())
			if test.is_first_of(freq):
				test.fast_algo_long()
				test.rebalance(0.02)
			test.update_next()
		print("Final Value: " + str(test.get_value()))
		print("Final Alloc: " + str(test.get_asset_alloc()))
		test.histogram(60)
		self.results_df["Date"] = dates
		self.results_df[test_name] = value

	def plot(self):
		# Extracting dates and portfolio values
		dates = self.results_df['Date']
		portfolio_columns = self.results_df.columns[1:]  # Exclude the 'Date' column
		portfolio_values = self.results_df[portfolio_columns]
		
		# Plotting
		plt.figure(figsize=(10, 6))
		for column in portfolio_values.columns:
			plt.plot(dates, portfolio_values[column], label=column)
		
		# Adding labels and title
		plt.xlabel('Date')
		plt.ylabel('Portfolio Value')
		plt.title('Portfolio Values Over Time')
		plt.legend()
		
		# Adjusting x-axis tick marks for better readability
		num_dates = len(dates)
		num_ticks = 10  # You can adjust the number of ticks as needed
		tick_step = max(num_dates // num_ticks, 1)
		plt.xticks(np.arange(0, num_dates, tick_step), rotation=45)
		
		# Displaying the plot
		plt.tight_layout()  # Ensures labels are not cut off
		plt.show()

	def save(self, filename):
		# Saving to CSV
		self.results_df.to_csv(filename, index=False)
		print(f"Portfolio values saved to {filename}")