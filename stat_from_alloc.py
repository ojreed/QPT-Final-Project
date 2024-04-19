import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def calculate_combined_stat(asset_allocation_list, beta_csv):
    # Load beta CSV
    beta_df = pd.read_csv(beta_csv, index_col='Date')

    # Initialize an empty DataFrame to store the combined data
    combined_df = pd.DataFrame(index=beta_df.index)
    
    # Iterate through each asset allocation
    for asset_allocation_csv in asset_allocation_list:
        # Extract file name from the file path
        file_name = os.path.basename(asset_allocation_csv).split('.')[0]
        
        # Load asset allocation CSV
        asset_allocation_df = pd.read_csv(asset_allocation_csv, index_col='Date')
        
        # Calculate combined value for each asset allocation and add to combined_df
        combined_df[file_name] = (asset_allocation_df * beta_df).sum(axis=1)

    return combined_df

def plot_combined_values(combined_df):
    plt.figure(figsize=(10, 5))
    for column in combined_df.columns:
        plt.plot(combined_df.index, combined_df[column], label=column)
    plt.title('Betas Over Time')
    plt.xlabel('Date')
    plt.ylabel('Daily Betas')
    plt.legend()
    plt.xticks(rotation=45)

    # Adjust x-axis labels to show only some of the dates
    num_dates = 10  # Number of dates to display
    n_ticks = len(combined_df.index) // num_dates
    plt.xticks(np.arange(0, len(combined_df.index), n_ticks), combined_df.index[::n_ticks], rotation=45)

    plt.tight_layout()
    plt.show()

# Example usage:
asset_allocation_list = [
    # "Allocations\\50-50 SPYxLQD - Daily Allocation.csv",
    "Allocations\\50-50 SPYxLQD - Monthly Allocation.csv",
    "Allocations\\50-50 SPYxLQD - Yearly Allocation.csv",
    "Allocations\\60-40 SPYxLQD - Daily Allocation.csv",
    "Allocations\\60-40 SPYxLQD - Monthly Allocation.csv",
    "Allocations\\60-40 SPYxLQD - Yearly Allocation.csv",
    "Allocations\\80-20 SPYxLQD - Daily Allocation.csv",
    "Allocations\\80-20 SPYxLQD - Monthly Allocation.csv",
    "Allocations\\80-20 SPYxLQD - Yearly Allocation.csv",
    "Allocations\\100 LQD Allocation.csv",
    "Allocations\\100 SPY Allocation.csv",
    "Allocations\\Fast Algo - Daily Allocation.csv",
    "Allocations\\Fast Algo - Monthly Allocation.csv",
    "Allocations\\Fast Algo - Yearly Allocation.csv",
    "Allocations\\Fast Algo Lintner - Daily Allocation.csv",
    "Allocations\\Fast Algo Lintner - Monthly Allocation.csv",
    "Allocations\\Fast Algo Lintner - Yearly Allocation.csv",
    "Allocations\\Round Robin - Daily Allocation.csv"
]
beta_csv = "Asset_Betas.csv"

result_df = calculate_combined_stat(asset_allocation_list, beta_csv)

print("Mean Value for Each List of Asset Allocation:")
print(result_df.mean())

plot_combined_values(result_df)
