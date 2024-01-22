# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:55:20 2024

@author: Rhino

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import cluster_tools as ct
import scipy.optimize as opt
import errors as err
import warnings

warnings.filterwarnings('ignore')


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


def one_silhouette(df_norm, n):
    """
    Function for calculating silhouette score for kmeans clustering
    Arguments: [int, DataFrame] Number of clusters, Normalized dataframe
    Returns:
    """
    # Set up the clusterer
    kmeans = cluster.KMeans(n_clusters=n, n_init=20)
    # Fit the data
    kmeans.fit(df_norm)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    # Calculate silhouette score
    score = skmet.silhouette_score(df_norm, labels)
    return score


def poly(x, a, b, c):
    x = x - 2003
    f = a + b*x + c*x**2

    return f


def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth
    rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))

    return f


def forecast(uk_df, variable):
    # Calculates the parameters and covariance
    param, covar = opt.curve_fit(logistic, uk_df["Year"],
                                 uk_df[variable], p0=(3e12, 4, 2004))
    print(param)
    # create array for forecasting
    year = np.linspace(1989, 2030, 100)
    forecast = logistic(year, *param)
    sigma = err.error_prop(year, logistic, param, covar)
    up = forecast + sigma
    low = forecast - sigma
    plt.figure()
    plt.plot(uk_df["Year"], uk_df[variable], label="GDP per capita")
    plt.plot(year, forecast, label="forecast")
    plt.fill_between(year, low, up, color="yellow", alpha=0.7)
    plt.xlabel("year")
    plt.ylabel("GDP")
    plt.legend()
    plt.show()


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
uk_df = uk_df.loc['1989':'2014']

# Checking correlation: Results show low correlation
print(uk_df.corr().round(3))
ct.map_corr(uk_df)
arr = pd.plotting.scatter_matrix(uk_df, figsize=(10, 10))

# Normalize values for clustering
uk_df_norm, uk_df_min, uk_df_max = ct.scaler(uk_df)

# Get the silhouette score for number of clusters
print()
for i in range(2, 11):
    score = one_silhouette(uk_df_norm, i)
    print(f"The silhouette score for {i: 3d} is {score: 7.4f}")

# Set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=4, n_init=20)
# Fit the data, results are stored in the kmeans object
kmeans.fit(uk_df_norm)
# Extract cluster labels
labels = kmeans.labels_
# Extract the estimated cluster centres and convert to original scales
cen = kmeans.cluster_centers_
cen = ct.backscale(cen, uk_df_min, uk_df_max)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]
# extract x and y values of data points
x = uk_df["electricity_kwh"]
y = uk_df["gdp_per_capita"]
plt.figure(figsize=(8.0, 8.0))
# Plot data with kmeans cluster number
cm = 'Set1'
plt.scatter(x, y, 10, labels, marker="o", cmap=cm)
# Plot cluster centres
plt.scatter(xkmeans, ykmeans, 45, "k", marker="d")
plt.xlabel("Electricity Consumption")
plt.ylabel("GDP per capita")
plt.show()


# Cluster plot gdp vs electricity consumed for UK
plt.figure(figsize=(10, 10), dpi=300)
plt.plot(uk_df.index, uk_df['electricity_kwh'])
plt.xlabel("Years", fontsize=14)
plt.ylabel("Electricity consumption per kwh", fontsize=13)
plt.title("Electricity Consumption in the UK", fontsize=18)
plt.xticks(ticks=['1989', '1994', '1999', '2004', '2009', '2014'],
           labels=['1989', '1994', '1999', '2004', '2009', '2014'])
plt.show()


plt.figure(figsize=(10, 10), dpi=300)
plt.plot(uk_df.index, uk_df["gdp_per_capita"])
plt.xlabel("Years", fontsize=14)
plt.ylabel("GDP per capita", fontsize=14)
plt.title("GDP per capita of the UK", fontsize=18)
plt.xticks(ticks=['1989', '1994', '1999', '2004', '2009', '2014'],
           labels=['1989', '1994', '1999', '2004', '2009', '2014'])
plt.show()

# Reset index and parse Year into a numeric value
uk_df = uk_df.reset_index()
uk_df["Year"] = pd.to_numeric(uk_df['Year'])

# Forecast GDP per capita
forecast(uk_df, 'gdp_per_capita')

forecast(uk_df, 'electricity_kwh')
