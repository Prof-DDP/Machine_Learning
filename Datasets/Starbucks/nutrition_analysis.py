# Tackling objective 2 (Design an easy-to-interpret visualization for the nutrition of each item) for Starbucks data
__author__ = 'Prof.D'

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('seaborn-whitegrid')

# Importing dataset
df = pd.read_csv(u'starbucks-menu-nutrition-food-updated.csv', encoding="ISO-8859-1")
df_numeric = df.drop(['FoodItem'], 1)

# Exploring variable correlations
def show_corr(df_, cmap_=False, cmap_colors=[]):
    if cmap_:
        plt.matshow(df_.astype(float).corr(), cmap=cmap_colors)
    else:
        plt.matshow(df_.astype(float).corr())
    plt.xticks(range(len(df_.columns)), df_.columns)
    plt.yticks(range(len(df_.columns)), df_.columns)
    plt.colorbar()
    
    plt.savefig('Color representations of variable correlations (Starbucks)', dpi=800)

    plt.show()

show_corr(df_numeric)

pd.plotting.scatter_matrix(df_numeric, figsize=(6,6))

plt.savefig('Scatter matrix showing variable correlations (Starbucks)', dpi=800)
plt.show()
#Fat seems to be a strong predictor of Calorie count (Carbs is good as well but not as good)

# Collecting data on certain foods
calories = df['Calories'].values
calories_sorted = sorted(calories, key=int, reverse=True)

top_ten_calorie_foods = [ df[ df['Calories'] == calorie_count] for calorie_count in calories_sorted[:10] ]
bottom_ten_calorie_foods = [ df[ df['Calories'] == calorie_count] for calorie_count in calories_sorted[-10:] ]

#Looking for common elements among selected foods
'''
columns = list(df_numeric.columns)
columns.remove('Calories')

avgs_top = {}
mins_top = {}
maxes_top = {}

avgs_bottom = {}
mins_bottom = {}
maxes_bottom = {}

for column in columns:
    column_vals_top = [food[column] for food in top_ten_calorie_foods]  
    avgs_top[column] = np.mean(column_vals_top)
    mins_top[column] = min(column_vals_top)
    maxes_top[column] = max(column_vals_top)
    
    column_vals_bottom = [food[column] for food in bottom_ten_calorie_foods] 
    avgs_bottom[column] = np.mean(column_vals_bottom)
    mins_bottom[column] = min(column_vals_bottom)
    maxes_bottom[column] = max(column_vals_bottom)
'''


