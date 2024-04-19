import Portfolio_Class as PF
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy import mean, std, sqrt


class Test_Class(object):
	"""docstring for Test_Class"""
	def __init__(self, Start_Date = (8,8,2013), Years = 5, Mode = 1):
		self.Start_Date = Start_Date
		self.Years = Years
		self.results_df = pd.DataFrame(columns=['Date'])
		self.mode = Mode
	
	def test_fixed_alloc(self,test_name,asset_list,Investments,freq = 1):
		alloc_df = pd.DataFrame(columns=['Date','SPY','XLE','XLF','XLK','XLV','XLI','XLY','XLP','XLU','XLB',"LQD","BIL","SHY","IEF","TLT","VGSH","VGIT","VGLT","BND"])
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
			new_row_data = {'Date':  test.get_date()}
			for asset, holding in test.holdings.items():
				new_row_data[asset] = holding
			for column in alloc_df.columns:
				if column not in new_row_data:
					new_row_data[column] = 0
			alloc_df = pd.concat([alloc_df,pd.DataFrame(new_row_data,index=[0])])
			test.update_next()
		# Save the CSV file to the folder
		file_name = str(test_name) + " Allocation.csv"
		file_path = os.path.join("Allocations", file_name)
		alloc_df.to_csv(file_path, index=False)
		print("Final Value: " + str(test.get_value()))
		print("Final Alloc: " + str(test.get_asset_alloc()))
		if self.mode == 1:
			test.histogram(100)
		self.results_df["Date"] = dates
		self.results_df[test_name] = value

	def test_fast_algo(self,test_name,asset_list,Investments,freq = 1):
		alloc_df = pd.DataFrame(columns=['Date','SPY','XLE','XLF','XLK','XLV','XLI','XLY','XLP','XLU','XLB',"LQD","BIL","SHY","IEF","TLT","VGSH","VGIT","VGLT","BND"])
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
				test.rebalance(0.004)
			new_row_data = {'Date':  test.get_date()}
			for asset, holding in test.holdings.items():
				new_row_data[asset] = holding
			for column in alloc_df.columns:
				if column not in new_row_data:
					new_row_data[column] = 0
			alloc_df = pd.concat([alloc_df,pd.DataFrame(new_row_data,index=[0])])
			test.update_next()
		# Save the CSV file to the folder
		file_name = str(test_name) + " Allocation.csv"
		file_path = os.path.join("Allocations", file_name)
		alloc_df.to_csv(file_path, index=False)
		print("Final Value: " + str(test.get_value()))
		print("Final Alloc: " + str(test.get_asset_alloc()))
		if self.mode == 1:
			test.histogram(100)
		self.results_df["Date"] = dates
		self.results_df[test_name] = value


	def test_fast_algo_S(self,test_name,asset_list,Investments,freq = 1):
		alloc_df = pd.DataFrame(columns=['Date','SPY','XLE','XLF','XLK','XLV','XLI','XLY','XLP','XLU','XLB',"LQD","BIL","SHY","IEF","TLT","VGSH","VGIT","VGLT","BND"])
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
				test.fast_algo('Lintner')
				test.rebalance(0.004)
			new_row_data = {'Date':  test.get_date()}
			for asset, holding in test.holdings.items():
				new_row_data[asset] = holding
			for column in alloc_df.columns:
				if column not in new_row_data:
					new_row_data[column] = 0
			alloc_df = pd.concat([alloc_df,pd.DataFrame(new_row_data,index=[0])])
			test.update_next()
		# Save the CSV file to the folder
		file_name = str(test_name) + " Allocation.csv"
		file_path = os.path.join("Allocations", file_name)
		alloc_df.to_csv(file_path, index=False)
		print("Final Value: " + str(test.get_value()))
		print("Final Alloc: " + str(test.get_asset_alloc()))
		if self.mode == 1:
			test.histogram(100)
		self.results_df["Date"] = dates
		self.results_df[test_name] = value
		print("SHARPE RATIO")
		print(mean(test.BM_return_history)/std(test.return_history)*sqrt(252))


	def test_RR(self,test_name,asset_list,Investments,test_window,quantile,freq = 1):
		alloc_df = pd.DataFrame(columns=['Date','SPY','XLE','XLF','XLK','XLV','XLI','XLY','XLP','XLU','XLB',"LQD","BIL","SHY","IEF","TLT","VGSH","VGIT","VGLT","BND"])
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
		test.calibrate_bull(test_window,quantile)
		dates = []
		value = []
		while not test.is_done():
			value.append(test.get_value())
			dates.append(test.get_date())
			if test.is_first_of(freq):
				test.Round_Robin(test_window)
			new_row_data = {'Date':  test.get_date()}
			for asset, holding in test.holdings.items():
				new_row_data[asset] = holding
			for column in alloc_df.columns:
				if column not in new_row_data:
					new_row_data[column] = 0
			alloc_df = pd.concat([alloc_df,pd.DataFrame(new_row_data,index=[0])])
			test.update_next()
		# Save the CSV file to the folder
		file_name = str(test_name) + " Allocation.csv"
		file_path = os.path.join("Allocations", file_name)
		alloc_df.to_csv(file_path, index=False)
		print("Final Value: " + str(test.get_value()))
		print("Final Alloc: " + str(test.get_asset_alloc()))
		if self.mode == 1:
			test.histogram(100)
			test.plot_ret()
		self.results_df["Date"] = dates
		self.results_df[test_name] = value

	def get_betas(self):
		return_tracker = pd.DataFrame(columns=['Date','SPY','XLE','XLF','XLK','XLV','XLI','XLY','XLP','XLU','XLB',"LQD","BIL","SHY","IEF","TLT","VGSH","VGIT","VGLT","BND"])
		alpha_tracker = pd.DataFrame(columns=['Date','SPY','XLE','XLF','XLK','XLV','XLI','XLY','XLP','XLU','XLB',"LQD","BIL","SHY","IEF","TLT","VGSH","VGIT","VGLT","BND"])
		beta_tracker = pd.DataFrame(columns=['Date','SPY','XLE','XLF','XLK','XLV','XLI','XLY','XLP','XLU','XLB',"LQD","BIL","SHY","IEF","TLT","VGSH","VGIT","VGLT","BND"])
		portfolios = []
		for asset in beta_tracker.columns:
			if asset != "Date":
				test = PF.Portfolio(pf_name=asset,asset_list=[asset],Investments=[100])
				test.set_start(self.Start_Date[0],self.Start_Date[1],self.Start_Date[2])
				test.set_end_relative(self.Years)
				portfolios.append(test)
		i = 0
		while not portfolios[0].is_done():
			ret_row = {'Date':  test.get_date()}
			alpha_row = {'Date':  test.get_date()}
			beta_row = {'Date':  test.get_date()}
			for test in portfolios:
				if asset != "Date":
					if i > 252:
						res = test.sharpe_regression()
						ret_row[test.pf_name] = res[0]["return"][0]
						alpha_row[test.pf_name] = res[0]["alpha"][0]
						beta_row[test.pf_name] = res[0]["beta"][0]
				test.update_next()
			if i > 252:
				return_tracker = pd.concat([return_tracker,pd.DataFrame(ret_row,index=[0])])
				alpha_tracker = pd.concat([alpha_tracker,pd.DataFrame(alpha_row,index=[0])])
				beta_tracker = pd.concat([beta_tracker,pd.DataFrame(beta_row,index=[0])])
			i+=1
		return_tracker.to_csv("Asset_Returns.csv", index=False)	
		alpha_tracker.to_csv("Asset_Alphas.csv", index=False)	
		beta_tracker.to_csv("Asset_Betas.csv", index=False)	
		
		# Plot asset beta over time
		plt.figure(figsize=(10, 5))
		for asset in beta_tracker.columns[1:]:
			plt.plot(beta_tracker['Date'], beta_tracker[asset], label=asset)
		plt.title('Asset Beta Over Time')
		plt.xlabel('Date')
		plt.ylabel('Beta')
		plt.legend()
		plt.xticks(rotation=45)

		# Adjust x-axis labels to show only some of the dates
		num_dates = 10  # Number of dates to display
		plt.xticks(
			range(0, len(beta_tracker['Date']), len(beta_tracker['Date']) // num_dates),
			beta_tracker['Date'][::len(beta_tracker['Date']) // num_dates], rotation=45
		)

		plt.tight_layout()
		plt.show()

		# Plot asset alpha over time
		plt.figure(figsize=(10, 5))
		for asset in alpha_tracker.columns[1:]:
			plt.plot(alpha_tracker['Date'], alpha_tracker[asset], label=asset)
		plt.title('Asset Alpha Over Time')
		plt.xlabel('Date')
		plt.ylabel('Alpha')
		plt.legend()
		plt.xticks(rotation=45)

		# Adjust x-axis labels to show only some of the dates
		plt.xticks(
			range(0, len(alpha_tracker['Date']), len(alpha_tracker['Date']) // num_dates),
			alpha_tracker['Date'][::len(alpha_tracker['Date']) // num_dates], rotation=45
		)

		plt.tight_layout()
		plt.show()

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