# Generating new products based on given infomation w/ markov chains

__author__ = 'Prof.D' #(5/27/18)

# Importing core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('seaborn-whitegrid')

# --- Food --- #

# Generating product names
from markov_chains import markov_generator

df = pd.read_csv(u'starbucks-menu-nutrition-food-updated.csv', encoding="ISO-8859-1")
food_names = df['FoodItem'].values

avg_food_name_len = int(np.array([len(name) for name in food_names]).mean())

def generate_and_check(source_txt, order, n_times):
    product = markov_generator(source_txt, order, n_times)
    if product in source_txt:
        check_result = True
    else:
        check_result = False
    return product, check_result

order = 5 #Formula for order could be: (avg name len / largest divsor of avg name len (exculding itself) ) + 2 OR (avg name len / smallest divisor of avg name len (excluding 1)) - 4
n_times = 100
product_name, check_result = generate_and_check(food_names, order, n_times)

#Favorite generated name so far: Everybody's Favorite - Bantam Bagels (2 Pack)

# Attempting to generate entrie products (Names + nutritional information)
foods = [list(df.iloc[i, :].values) for i in range(len(df))]
for i,food in enumerate(foods):
    for j,val in enumerate(food):
        food[j] = str(val)
    foods[i] = ','.join(food)

avg_full_food_name_len = int(np.array([len(food) for food in foods]).mean())

highest_divisor = 15 #Highest divisor of avg_full_food_name_len
lowest_divisor = 3

order_formula_high = int((avg_full_food_name_len / highest_divisor) + 2)
order_formula_low = int((avg_full_food_name_len / lowest_divisor) - 4)

ofh = order_formula_high
ofl = order_formula_low
#order = 5
#n_times = 100
product_name, check_result = generate_and_check(foods, ofh, n_times)

def check(product_name):
    pn = product_name.split(',')
    if pn[0] in food_names:
        check_result_1 = False
    else:
        check_result_1 = True
    
    if len(pn) == 6:
        nums = pn[1:]
        for num in nums:
            if num.isdigit():
                check_result_2 = True
            else:
                check_result_2 = False
    else:
        check_result_2 = False
    
    return check_result_1, check_result_2

def generate_and_check2(n):
    approved_products_ofh = []
    total_ofh = 0
    
    approved_products_ofl = []
    total_ofl = 0
    
    approved_products_rand = []
    total_rand = 0
    
    for i in range(n):
        product = markov_generator(foods, ofh, n_times)
        check_1, check_2 = check(product)
        if check_1 == True and check_2 == True:
            approved_products_ofh.append(product)
        total_ofh+=1
    approved_rate_ofh = len(approved_products_ofh) / total_ofh
    
    for i in range(n):
        product = markov_generator(foods, ofl, n_times)
        check_1, check_2 = check(product)
        if check_1 == True and check_2 == True:
            approved_products_ofl.append(product)
        total_ofl+=1
    approved_rate_ofl = len(approved_products_ofl) / total_ofl
    
    rand_order = np.random.randint(4, avg_full_food_name_len-4)
    if rand_order == ofh or rand_order == ofl:
        rand_order = np.random.randint(4, avg_full_food_name_len-4)
    
    for i in range(n):
        product = markov_generator(foods, rand_order, n_times)
        check_1, check_2 = check(product)
        if check_1 == True and check_2 == True:
            approved_products_rand.append(product)
        total_rand+=1
    approved_rate_rand = len(approved_products_rand) / total_rand
    
    return approved_products_ofh, total_ofh, approved_rate_ofh, approved_products_ofl, total_ofl, approved_rate_ofl, approved_products_rand, total_rand, approved_rate_rand, rand_order

rates_ofh = []
rates_ofl = []
rates_rand = []
rands = []

for i in range(100):
    approved_products_ofh, total_ofh, approved_rate_ofh, approved_products_ofl, total_ofl, approved_rate_ofl, approved_products_rand, total_rand, approved_rate_rand, rand_order = generate_and_check2(1000)
    rates_ofh.append(approved_rate_ofh)
    rates_ofl.append(approved_rate_ofl)
    rates_rand.append(approved_rate_rand)
    rands.append(rand_order)
    
    if i == 25:
        print('1/4 of the way there!')
    elif i == 50:
        print('Halfway there!')
    elif i == 75:
        print('3/4 done!')
        
avg_rate_ofh = np.array(rates_ofh).mean()
avg_rate_ofl = np.array(rates_ofl).mean()
avg_rate_rand = np.array(rates_rand).mean()


# Plotting relationship between results of random rates and rates
fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(rands, rates_rand, color='orange', alpha=0.8)

ax.set_xlabel('Randomly generated orders')
ax.set_ylabel('Resulting approved product rate')
ax.set_title('Random orders vs. Approved product rates')
ax.legend()

plt.show()

# Very Interesting relationship. Testing with all numbers within sample range.
x = list(np.arange(4, avg_full_food_name_len-4))
x.remove(ofh)
x.remove(ofl)

avgs = []
rates = []
approved_products = []
total = 0

for num in x:
    for j in range(100):
        for i in range(100):
            product = markov_generator(foods, num, n_times)
            check_1, check_2 = check(product)
            if check_1 == True and check_2 == True:
                approved_products.append(product)
            total+=1
        approved_rate = len(approved_products) / total
        rates.append(approved_rate)
    avgs.append(np.array(rates).mean())
    rates = []    
    total = 0
    approved_products = []
    
    if num == 9:
        print('1/4 of the way there!')
    elif num == 18:
        print('Halfway there!')
    elif num == 26:
        print('3/4 done!')

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(x, avgs, color='orange', alpha=0.8)
ax.scatter([ofh, ofl], [avg_rate_ofh, avg_rate_ofl], color='blue', alpha=0.8)

ax.set_xlabel('Orders')
ax.set_ylabel('Resulting approved product rate')
ax.set_title('Orders vs. Approved product rates')
ax.legend()

plt.show()

# Testing on different dataset. Curious to see if same results will occur


        
            




