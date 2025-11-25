import pandas as pd

# Path to the CSV file
csv_file = "data/GraphAllocBench-v2/summary_model_step_1000000.csv"

# Read the CSV file
data = pd.read_csv(csv_file)

# Group by 'env_name' and calculate the mean for each numeric column
averages = data.groupby('env_name').mean(numeric_only=True)

# Save the results to a new CSV file
output_file = "data/GraphAllocBench-v2/averages_summary.csv"
averages.to_csv(output_file)

print(f"Averages calculated and saved to {output_file}")