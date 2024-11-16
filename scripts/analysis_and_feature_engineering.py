import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path):
  df = pd.read_csv(path)
  return df

def data_description(data):
  print(data.head())
  print(data.info())
  print(data.isnull().sum())
  print(data.describe())

def plot_catagorical_column(categorical_columns, data):
  for col in categorical_columns:
    plt.figure(figsize=(10, 4))
    sns.countplot(x=col, data=data , color= 'red')
    plt.title(f'Distribution for {col}')
    plt.xticks(rotation=90)
    plt.show()

def plot_numeric_column(numeric_columns, df):
  for col in numeric_columns:
    plt.figure(figsize=(10, 4))
    sns.histplot(df[col], kde=True , color="brown")
    plt.title(f'Distribution for {col}')
    plt.show()

def correlation_matrix(df, numeric_columns):
  corr = df[numeric_columns].corr()
  # Generate a mask for the upper triangle
  mask = np.triu(np.ones_like(corr, dtype=bool))
  # Set up the matplotlib figure
  f, ax = plt.subplots(figsize=(11, 9))
  # Generate a custom diverging colormap
  cmap = sns.diverging_palette(230, 20, as_cmap=True)
  # Draw the heatmap with the mask and correct aspect ratio
  sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
              square=True, linewidths=.5, cbar_kws={"shrink": .5} , color= 'purple')
  plt.title('Correlation Heatmap')
  plt.show()

def outliers(df, numeric_columns):
  for col in numeric_columns:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df[col], color= 'orange')
    plt.title(f'Box Plot for {col}')
    plt.show()

def save_data(df, path):
  df.to_csv(path, index=False)