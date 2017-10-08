import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Implementing UCB
import math
N = 10000  # total iterations
d = 10  # number of ad's
ads_selected = []
number_of_selections =[0] * d
sums_of_rewards = [0] * d
total_reward = 0
for n in range(0, N):
    max_upper_bound = 0
    ad = 0
    for i in range(0, d):
        if(number_of_selections[i] > 0):   # only use this after the first 10 iterations
            average_reward = sums_of_rewards[i] / number_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / number_of_selections[i])
            upper_bound = average_reward + delta_i
        else:   # for first 10 iterations just use all 10 ad one by one for each round
            upper_bound = 1e400  # 10 to the power of 400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1
    
    # In real world we would get actual user inputs but here we are using our dataset as simulated user inputs
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
    
# Visualising UCB
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
    
        
        
        