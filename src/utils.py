import seaborn as sns
from scipy.stats import linregress
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_scatterplot(df, x_col, y_col, title, xlabel, ylabel):
    """
    This function creates a scatter plot with a linear regression line from a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    x_col (str): The column in the DataFrame to use for the x-axis.
    y_col (str): The column in the DataFrame to use for the y-axis.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    """

    # Create the plot
    plt.figure(figsize=(7, 7))
    sns.regplot(x=df[x_col], y=df[y_col], scatter_kws={"alpha": 0.3})

    # Add labels and title
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Show the plot
    plt.show()

def kepler_repair_data(df, df_zones):
    # Add the lat and lng from Pick up location id
    df_parsed = pd.merge(df, df_zones[['LocationID', 'lat', 'lng']], left_on='PULocationID', right_on='LocationID',
                       how='left')

    # Rename lat and lng so we can add then again for DOLocationID
    df_parsed.rename(columns={'lat': 'lat_PU', 'lng': 'lng_PU'}, inplace=True)

    # Add the lat and lng from Drop off location id
    df_parsed = pd.merge(df_parsed, df_zones[['LocationID', 'lat', 'lng']], left_on='DOLocationID', right_on='LocationID',
                       how='left')

    return df_parsed


def plot_scatter_with_trendline(ax, x, y, title, xlabel, ylabel, color, xlim=None):
    sns.regplot(x=x, y=y, ax=ax, scatter_kws={'alpha': 0.3, 'color': color}, line_kws={'color': 'red'})
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend([f'y={intercept:.2f}+{slope:.2f}x\nR²={r_value ** 2:.2f}'])
    if xlim:
        ax.set_xlim(xlim)
    return slope, intercept, r_value, p_value, std_err


def remove_outliers(df, min_trip_distance=0, max_trip_distance=1000, min_fare=0, max_fare=1000):
  """
  This function removes outliers from a DataFrame containing trip data based on trip distance and fare amount.

  Args:
      df: A pandas DataFrame containing trip data with columns for "trip_distance" and "fare_amount".
      max_trip_distance (optional): The maximum allowed trip distance (default: 1000).
      min_fare (optional): The minimum allowed fare amount (default: 0).
      max_fare (optional): The maximum allowed fare amount (default: 1000).
      min_trip_distance (optional): The minimum allowed trip distance (default: 0).

  Returns:
      A new DataFrame containing only rows where trip distance falls within the specified range and fare amount is within the specified bounds.
  """

  return df[(df["trip_distance"] > min_trip_distance) & (df["trip_distance"] < max_trip_distance) & 
            (df["fare_amount"] > min_fare) & (df["fare_amount"] < max_fare)]


def get_a_random_chunk_property(data):
    """
    This function only serves an example of fetching some of the properties
    from the data.
    Indeed, all the content in "data" may be useful for your project!
    """

    chunk_index = np.random.choice(len(data))

    date_list = list(data[chunk_index]["near_earth_objects"].keys())

    date = np.random.choice(date_list)

    objects_data = data[chunk_index]["near_earth_objects"][date]

    object_index = np.random.choice(len(objects_data))

    object = objects_data[object_index]

    properties = list(object.keys())
    property = np.random.choice(properties)

    print("date:", date)
    print("NEO name:", object["name"])
    print(f"{property}:", object[property])


def load_data_from_google_drive(url):
    url_processed='https://drive.google.com/uc?id=' + url.split('/')[-2]
    df = pd.read_csv(url_processed)
    return df