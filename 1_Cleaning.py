import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

df=pd.read_csv("/content/IRIS.csv")
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values
df_cleaned =df.drop(columns=['species'])

print(df_cleaned)

# Fill missing values with mean or other strategies
df_filled = df.fillna(df.mean())

# Check for duplicates
print(df.duplicated().sum())

# Remove duplicates
df_no_duplicates = df.drop_duplicates()

# Check data types
print(df_cleaned.dtypes)

# Data Analysis
print(df_cleaned.describe())

# Calculate correlation matrix
correlation_matrix = df_cleaned.corr()

# Visualize correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
plt.matshow(correlation_matrix, cmap='viridis')
plt.title('Correlation Matrix')
plt.colorbar()
plt.show()
# Perform t-test or other statistical tests
t_stat, p_value = stats.ttest_ind(df_cleaned['group1'], df_cleaned['group2'])
print(f'T-statistic: {t_stat}, p-value: {p_value}')
# Plot histogram
plt.hist(df_cleaned['sepal_length'], bins=20, color='blue', alpha=0.7)
plt.title('Histogram of sepal_length')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

# Scatter plot
plt.scatter(df['petal_length'], df['petal_width'], color='red', alpha=0.5)
plt.title('Scatter Plot between petal_length and petal_width')
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.show()

