from sklearn.preprocessing import StandardScaler
import numpy as np

data = [[2,40],[4,50],[-2,60],[-4,70]]

print(f"data pre processing:\n{data}")
scaler = StandardScaler()
scaler.fit(data)

print(f'scaler.mean_ pre processing: {scaler.mean_}')
print(f'scaler.var pre processing: {scaler.var_}')

# mean normalization + feature scaling (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler.transform)
# Standardize features by removing the mean and scaling to unit variance
# The standard score of a sample x is calculated as:
# z = (x - u) / s
# where u is the mean of the training samples or zero if with_mean=False, 
# and s is the standard deviation of the training samples or one if with_std=False.

data_post_processed = scaler.transform(data)
print(f"data post processing:\n{data_post_processed}")
scaler.fit(data_post_processed)
print(f'scaler.mean_ post processing\n{scaler.mean_}')
print(f'scaler.var post processing: {scaler.var_}')

