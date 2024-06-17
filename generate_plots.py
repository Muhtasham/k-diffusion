import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the results CSV file with the appropriate encoding
data = pd.read_csv('results.csv', encoding='utf-8')

# Calculate MFU if not present
if 'MFU' not in data.columns or data['MFU'].isnull().any():
    theoretical_peak_flops = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
    data['MFU'] = data.apply(lambda row: row['Total FLOPs'] / (row['Elapsed Time (s)'] * theoretical_peak_flops), axis=1)

# Set the style for the plots
sns.set_theme(style="whitegrid")

# Convert compile flag to string for better plotting
data['Compile Flag'] = data['Compile Flag'].apply(lambda x: 'Compile' if '--compile' in str(x) else 'No Compile')

# Plot MFU
plt.figure(figsize=(12, 8))
sns.lineplot(data=data, x='Batch Size', y='MFU', hue='Config', style='Compile Flag', markers=True, dashes=False, palette='Set1')
plt.title('MFU vs. Batch Size', fontsize=16)
plt.xlabel('Batch Size', fontsize=14)
plt.ylabel('MFU', fontsize=14)
plt.legend(title='Configuration and Compile Flag', fontsize=12)
plt.xticks(data['Batch Size'].unique())
plt.grid(True)
plt.tight_layout()
plt.savefig('mfu_vs_batch_size.png')
plt.show()

# Plot FLOPs per second
plt.figure(figsize=(12, 8))
data['FLOPs per Second'] = data['Total FLOPs'] / data['Elapsed Time (s)']
sns.lineplot(data=data, x='Batch Size', y='FLOPs per Second', hue='Config', style='Compile Flag', markers=True, dashes=False, palette='Set2')
plt.title('FLOPs per Second vs. Batch Size', fontsize=16)
plt.xlabel('Batch Size', fontsize=14)
plt.ylabel('FLOPs per Second', fontsize=14)
plt.legend(title='Configuration and Compile Flag', fontsize=12)
plt.xticks(data['Batch Size'].unique())
plt.grid(True)
plt.tight_layout()
plt.savefig('flops_per_second_vs_batch_size.png')
plt.show()
