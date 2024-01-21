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
    df = pd.read_csv(path, skiprows=4, index_col=0)
    # In this case we don't need to transpose our data
    # If we had to, then we could use df = df.T
    # Drop the unnamed column
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # Drop unwanted columns
    df.drop(columns=['Indicator Code', 'Country Code',
                     'Indicator Name'], inplace=True)

    return df


df = read('electric.csv')
print(df)
