# GeMS 
d = [16,32]
beta = [0.1, 0.2, 0.5, 1.0, 2.0]
gamma = [0.1, 0.2, 0.5, 1.0]

# SAC
learning_rate_q_values = [0.0001, 0.0005, 0.001, 0.002, 0.005]
learning_rate_policy_values = [0.0001, 0.0005, 0.001, 0.003, 0.005, 0.01]
discount_factor_values = [0.6, 0.7, 0.8, 0.9, 0.99]
polyak_averaging_values = [0.001, 0.002, 0.005, 0.01]

print("Number of grid seach combinations for GeMS is: ", len(d)*len(beta)*len(gamma))
print("Number of grid seach combinations for SAC is: ", len(learning_rate_q_values)*len(learning_rate_policy_values)*len(discount_factor_values)*len(polyak_averaging_values))