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
plt.rc("axes.spines", top=False, right=False)


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
    Returns: [float] score
    """
    # Set up the clusterer
    kmeans = cluster.KMeans(n_clusters=n, n_init=20)
    # Fit the data
    kmeans.fit(df_norm)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    print(centers)
    # Calculate silhouette score
    score = skmet.silhouette_score(df_norm, labels)
    return score


def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth
    rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))

    return f


# Define the function for the cubic fit
def cubic_fit(x, a, b, c, d):
    """
    Calulcates the cubic function to fit the graph
    Arguments: Function parameters
    Returns: Function value
    """
    return a * x**3 + b * x**2 + c * x + d


# Define the combined function (linear + sine)
def combined_function(x, a, b, c, amplitude, frequency, phase):
    """
    Calculates a combined polynomial and sin function to fit the graph
    Arguments: Function parameters
    Returns: Function value
    """
    return a * x + b + amplitude * np.sin(2 * np.pi * frequency * (x - phase))


def forecast(df, column, country, f, n=0,):
    """
    Takes five arguments and plots a forecast curve using curve_fit
    Arguments: Dataframe, Column name, n, f can be 1,2 or 3 indicating curve
    functions logistic, combined_function and cubic_fit respectively.
    Return: None
    """
    # If f is 1 then call logistic function
    if f == 1:
        # Calculates the parameters and covariance
        param, covar = opt.curve_fit(logistic, df["Year"], df[column],
                                     maxfev=4000, p0=(3e12, n, 2004))
        year = np.linspace(1989, 2024, 1000)
        # create array for forecasting
        forecast = logistic(year, *param)
        # Calculcate the sigma value
        sigma = err.error_prop(year, logistic, param, covar)
        up = forecast + sigma
        low = forecast - sigma
        plt.figure(figsize=(12, 10), dpi=400)
        plt.plot(df["Year"], df[column], label=column)
        plt.plot(year, forecast, label="Forecast")
        plt.fill_between(year, low, up, color="yellow", alpha=0.7,
                         label='Confidence margin')
        plt.xlabel("Year", fontsize=15)
        plt.ylabel(column.title(), fontsize=15)
        if column == 'gdp_per_capita':
            plt.title('GDP per capita forecast', fontsize=18)
        else:
            plt.title('Electricity Consumption per capita', fontsize=18)
        plt.legend(loc=4, fontsize=15)
        plt.savefig(f'{country}_{column}.png', dpi=400)
        plt.show()

    elif f == 2:  # If f is 2 call combined function
        # Provide initial guesses for the parameters
        initial_guesses = [1, 5, 10, 5000, 0.1, 1990]
        # Set bounds for the parameters
        bounds = ([-np.inf, -np.inf, -np.inf, 0, 0, -np.inf],
                  [np.inf, np.inf, np.inf, 100000, np.inf, np.inf])
        # Perform the curve fitting with increased maxfev and initial guesses
        param, covar = opt.curve_fit(combined_function, df['Year'],
                                     df[column], maxfev=3000,
                                     p0=initial_guesses, bounds=bounds)
        # Extract the optimized parameters
        a, b, c, amplitude, frequency, phase = param
        year = np.linspace(1989, 2024, 1000)
        # Generate the fitted curve
        forecast = combined_function(year, a, b, c, amplitude,
                                     frequency, phase)
        # Calculcate the sigma value
        sigma = err.error_prop(year, combined_function, param, covar)
        # This is causing problems.... curve_fit cant get covariance array
        # values correct
        print(covar)
        up = forecast + sigma
        low = forecast - sigma
        plt.figure(figsize=(12, 10), dpi=400)
        plt.plot(df["Year"], df[column], label=column)
        plt.plot(year, forecast, label="Forecast")
        plt.fill_between(year, low, up, color="yellow", alpha=0.7,
                         label='Confidence margin')
        plt.xlabel("Year", fontsize=15)
        plt.ylabel(column.title(), fontsize=15)
        if column == 'gdp_per_capita':
            plt.title('GDP per capita forecast', fontsize=18)
        else:
            plt.title('Electricity Consumption per capita', fontsize=18)
        plt.legend(loc=4, fontsize=15)
        plt.savefig(f'{country}_{column}.png', dpi=400)
        plt.show()

    elif f == 3:  # If f is 3 then call cubic function
        # Perform the curve fitting
        param, covar = opt.curve_fit(cubic_fit, df['Year'],
                                     df[column])

        # Extract the optimized parameters
        a, b, c, d = param
        year = np.linspace(1989, 2019, 1000)
        # Generate the fitted curve
        forecast = cubic_fit(year, a, b, c, d)
        # Calculcate the sigma value
        sigma = err.error_prop(year, cubic_fit, param, covar)
        up = forecast + sigma
        low = forecast - sigma
        plt.figure(figsize=(12, 10), dpi=400)
        plt.plot(df["Year"], df[column], label=column)
        plt.plot(year, forecast, label="Forecast")
        plt.fill_between(year, low, up, color="yellow", alpha=0.7,
                         label='Confidence margin')
        plt.xlabel("Year", fontsize=15)
        plt.ylabel(column.title(), fontsize=15)
        if column == 'gdp_per_capita':
            plt.title('GDP per capita forecast', fontsize=18)
        else:
            plt.title('Electricity Consumption per capita', fontsize=18)
        plt.legend(loc=4, fontsize=15)
        plt.savefig(f'{country}_{column}.png', dpi=400)
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
plt.scatter(x, y, 40, labels, marker="o", cmap=cm)
# Plot cluster centres
plt.scatter(xkmeans, ykmeans, 65, "k", marker="d")
plt.xlabel("Electricity Consumption")
plt.ylabel("GDP per capita")
plt.title('Electricity consumption vs GDP per capita', fontsize=18)
plt.savefig('cluster.png', dpi=400)
plt.show()


# Plot for electricity consumption in the UK: 1989-2014
plt.figure(figsize=(10, 10), dpi=300)
plt.plot(uk_df.index, uk_df['electricity_kwh'])
plt.xlabel("Years", fontsize=14)
plt.ylabel("Electricity consumption per kwh", fontsize=13)
plt.title("Electricity Consumption in the UK", fontsize=18)
plt.xticks(ticks=['1989', '1994', '1999', '2004', '2009', '2014'],
           labels=['1989', '1994', '1999', '2004', '2009', '2014'])
plt.savefig('UK_kwh.png', dpi=400)
plt.show()

# Plot for gdp per capita in the UK: 1989-2014
plt.figure(figsize=(10, 10), dpi=300)
plt.plot(uk_df.index, uk_df["gdp_per_capita"])
plt.xlabel("Years", fontsize=14)
plt.ylabel("GDP per capita", fontsize=14)
plt.title("GDP per capita of the UK", fontsize=18)
plt.xticks(ticks=['1989', '1994', '1999', '2004', '2009', '2014'],
           labels=['1989', '1994', '1999', '2004', '2009', '2014'])
plt.savefig('UK_gdp_line.png', dpi=400)
plt.show()

# Reset index and parse Year into a numeric value
uk_df = uk_df.reset_index()
uk_df["Year"] = pd.to_numeric(uk_df['Year'])

# Forecast GDP per capita
forecast(uk_df, 'gdp_per_capita', 'UK', 1, 4)

# Forecast Electricity consumption per capita
forecast(uk_df, 'electricity_kwh', 'UK', 3)

# Get data for Country of interest and merge it
germany_df = pd.merge(electric_df['Germany'], gdp_df['Germany'],
                      on=electric_df.index)
# Rename Columns
germany_df.rename(columns={'key_0': 'Year', 'Germany_x': 'electricity_kwh',
                  'Germany_y': 'gdp_per_capita'}, inplace=True)
germany_df.set_index('Year', inplace=True)
# Slice the data from 1990-2014
germany_df = germany_df.loc['1989':'2014']

# Plot for gdp per capita of Germany: 1989-2014
plt.figure(figsize=(10, 10), dpi=300)
plt.plot(germany_df.index, germany_df["gdp_per_capita"])
plt.xlabel("Years", fontsize=14)
plt.ylabel("GDP per capita", fontsize=14)
plt.title("GDP per capita of Germany", fontsize=18)
plt.xticks(ticks=['1989', '1994', '1999', '2004', '2009', '2014'],
           labels=['1989', '1994', '1999', '2004', '2009', '2014'])
plt.savefig('Germany_gdp_line.png', dpi=400)
plt.show()

# Reset index and parse Year into a numeric value
germany_df = germany_df.reset_index()
germany_df["Year"] = pd.to_numeric(germany_df['Year'])

# Forecast for germany
forecast(germany_df, 'gdp_per_capita', 'Germany', 2)
