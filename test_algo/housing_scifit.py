#import findspark
#findspark.init()

import pandas as pd

datafile = pd.read_csv("/Users/jih108/spark_test/spark_my_summary/house_estimator/housing.csv")

print(datafile.head())
datafile.info()
print(datafile["ocean_proximity"].value_counts())
print(datafile.describe())

import matplotlib.pyplot as plt
import numpy as np

"""
plt.figure(1)
plt.subplot(121)
datafile["total_bedrooms"].hist(bins=50)
plt.subplot(122)
plt.scatter(datafile["total_bedrooms"], datafile["total_rooms"])
plt.show()
"""

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

train_set, test_set = train_test_split(datafile, test_size=0.2, random_state=42)

print("train set => ", train_set.describe())

print("test set => ", test_set.describe())

#plt.figure(1)
#plt.subplot(221)
#train_set.hist(bins=50)
datafile["income_cat"] = np.ceil(datafile["median_income"]/1.5)
datafile["income_cat"].where(datafile["income_cat"] < 5, 5.0, inplace=True)
#plt.hist(datafile["income_cat"])
#plt.show()

#print("total length = ", len(datafile))
#print(datafile["income_cat"].value_counts(), len(datafile), datafile["income_cat"].value_counts()/len(datafile))

#datafile.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
#plt.show()
datafile.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
              s=datafile["population"]/100, label="population", figsize=(10,7),
              c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
plt.show()


corr_matrix = datafile.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))
"""
plt.figure(1)
plt.subplot(211)
datafile.plot(kind="scatter", x="longitude", y="latitude")
plt.subplot(212)
plt.scatter(datafile["longitude"], datafile["latitude"], color='blue')
plt.show()
"""


from pandas.tools.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
pd.plotting.scatter_matrix(datafile[attributes], figsize=(12,8))
plt.show()


####scikit test
#housing_labels = datafile["median_house_value"].copy()
#print(housing_labels.head())

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

print(datafile.values)
