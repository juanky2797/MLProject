import matplotlib
import pandas as pd
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import random



pd.set_option('display.max_columns', None)
df = pd.read_csv('data/application_including_boundary_data.csv',encoding='latin-1')

to_drop = ['OBJECTID', 'ApplicationNumber', 'DevelopmentDescription']


print(df)

