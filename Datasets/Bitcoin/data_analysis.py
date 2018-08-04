# Data analysis and EDA on Bitcoin (market-price) dataset

__author__ = 'Prof.D'

# Importing core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('seaborn-whitegrid')
np.random.seed(42)

# Importing dataset
df = pd.read_csv(r'coindesk-bpi-USD-close_data-2010-07-17_2018-08-01.csv') # Contains daily close prices of bitcoin from its creation to current date

# Encoding Date variable
#Simple numeric encoding. Starts at day 0 and ends on day 2937
df['Date'] = np.arange(len(df['Date']))

# Features and labels
X = df['Date'].values
y = df['Close Price'].values

# Visualizing data
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(X, y, c='b')
ax.set_title('Closing Price of Bitcoin over time')
ax.set_xlabel('Day')
ax.set_ylabel('Closing Price')

plt.show()

#Data looks pretty stagnant in begining and middle. 
#Lets investigate some of the data's attributes
#For the data as a whole
min_price = min(y)
max_price = max(y)

std_of_prices = np.std(y)
mean_price = np.mean(y)

#For indiviual timespans
'''
value_thresholds = np.arange(100)
threshold_indices = [np.argmax(y > val) for val in value_thresholds]
print(threshold_indices)
'''
#Seeing how many days it took for bitcoin to reach certain prices. 
timespans = np.arange(0, 5100, 100)
n_days = {}

for i,timespan in enumerate(timespans):
    if i == 0:
        n_days[timespan] = (np.argmax(y >= timespan))
    else:
        n_days[timespan] = (np.argmax(y >= timespan) - list(n_days.values())[i-1])

max_value_gap_in_timespan = list(n_days.keys())[np.argmax(list(n_days.values()))]





