import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file to examine its structure
file_path = 'synapsecount.csv'

# Reload the data without headers and rename columns accordingly
data_no_headers = pd.read_csv(file_path, header=None)
data_no_headers.columns = ['Values', 'Binary']

# Configure font sizes and tick properties globally
plt.rcParams.update({
    'font.size': 16,  # Set global font size
    'xtick.labelsize': 14,  # X-axis tick size
    'ytick.labelsize': 14,  # Y-axis tick size
    'xtick.color': 'black',  # X-axis tick color
    'ytick.color': 'black',  # Y-axis tick color
    'xtick.major.size': 8,  # X-axis major tick size
    'ytick.major.size': 8,  # Y-axis major tick size
    'xtick.major.width': 1.5,  # X-axis major tick width
    'ytick.major.width': 1.5,  # Y-axis major tick width
    'axes.labelweight': 'bold',  # Bold labels
    'axes.labelsize': 18  # Label size
})

# Histogram for 'Values'
plt.figure(figsize=(8, 6))
plt.hist(data_no_headers['Values'], bins=15, alpha=0.7, edgecolor='black')
plt.xlabel('Vesicles per Synapse')
plt.ylabel('Frequency')
plt.show()

# Pie chart for 'Binary'
plt.figure(figsize=(6, 6))
binary_counts = data_no_headers['Binary'].value_counts()
plt.pie(
    binary_counts,
    labels=['No Post-Synapse', 'Possible Post-Synapse'],
    autopct='%1.1f%%',
    colors=['lightcoral', 'lightskyblue'],
    startangle=90,
    textprops={'fontsize': 14, 'weight': 'bold', 'color': 'black'}
)
plt.title('Fraction of No Post-Synapse and Possible Post-Synapse', fontsize=16, weight='bold')
plt.show()
