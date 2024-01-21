# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:55:20 2024

@author: Rhino

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import cluster_tools as ct
import errors as err


def read(path):
    """
    This function reads a csv file with pandas and takes path argument.

    Arguments: Takes a file path to read
    Returns: Returns a transposed dataframe object

    """
    df = pd.read_csv(path, skiprows=4, index_col=0)
    # Drop the unnamed column
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # Drop unwanted columns
    df.drop(columns=['Indicator Code', 'Country Code',
                     'Indicator Name'], inplace=True)
    # Clean data further, drop columns with NA values
    electric_df.dropna(axis=1, how='all', inplace=True)
    return df.T  # Return Transposed dataframe


def kmeans(n):
    """
    Function for kmeans clustering
    Arguments:
    Returns:
    """
    print('Silhoutte Scores:\n')
    for i in range(2, 11):
        kmeans = cluster.Kmeans(nclusters=n)
        kmeans.fit()


# Get our data from the files using the read function
electric_df = read('electric.csv')
print(electric_df)

gdp_df = read('gdp.csv')
print(gdp_df)

# Get data for Country of interest and merge it
uk_df = pd.merge(electric_df['United Kingdom'], gdp_df['United Kingdom'],
                 on=electric_df.index)
# Rename Columns
uk_df.rename(columns={'key_0': 'Year', 'United Kingdom_x': 'electricity_kwh',
                      'United Kingdom_y': 'gdp_per_capita'}, inplace=True)
uk_df.set_index('Year', inplace=True)
# Slice the data from 1990-2014
uk_df = uk_df.loc['1990':'2014']

# Checking correlation: Results show low correlation
print(uk_df.corr().round(3))
ct.map_corr(uk_df)
arr = pd.plotting.scatter_matrix(uk_df, figsize=(10, 10))

# Normalize values for clustering
uk_df_norm, uk_df_min, uk_df_max = ct.scaler(uk_df)

# Get the silhouette score for number of clusters
