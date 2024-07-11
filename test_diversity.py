from itertools import combinations

def diversity_reward(slate):
    distances = [jaccard_distance(i, j) for i, j in combinations(slate, 2)]
    return sum(distances) / (len(slate) * (len(slate) - 1))

def jaccard_distance(item1, item2):
    # define for each index whether the value is the same for both items
    same = [i == j for i, j in zip(item1, item2)]
    # calculate the Jaccard distance
    return 1 - sum(same) / len(same) 

list_of_vectors1 = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
list_of_vectors2 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
list_of_vectors3 = [[1, 1, 1], [0, 0, 0], [1, 1, 1]]
list_of_vectors4 = [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
list_of_vectors5 = [[1, 1, 0], [1, 1, 1], [1, 1, 1]]

for i in range(5):
    list_of_vectors = locals()['list_of_vectors' + str(i + 1)]    
    
    print(f"List of vectors {i + 1}: {list_of_vectors}")
    print(f"Diversity reward: {diversity_reward(list_of_vectors)}")

