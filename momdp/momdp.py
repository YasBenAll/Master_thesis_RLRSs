import numpy as np
from itertools import combinations
from collections import defaultdict

class MOMDP:
    def __init__(self, states, items, slate_size, gamma=1.0):
        self.states = states
        self.items = items
        self.slate_size = slate_size
        self.gamma = gamma
        self.actions = list(combinations(items, slate_size))
        self.transition_probabilities = defaultdict(lambda: defaultdict(float))
        self.reward_functions = {
            'engagement': self.engagement_reward,
            'diversity': self.diversity_reward,
            'novelty': self.novelty_reward
        }

    def engagement_reward(self, clicks):
        return sum(clicks)

    def diversity_reward(self, slate):
        distances = [self.jaccard_distance(i, j) for i, j in combinations(slate, 2)]
        return sum(distances) / (len(slate) * (len(slate) - 1))

    def novelty_reward(self, slate, item_popularity):
        return np.mean([-np.log2(item_popularity[i]) for i in slate])

    def jaccard_distance(self, item1, item2):
        set1 = set(item1)
        set2 = set(item2)
        return 1 - len(set1 & set2) / len(set1 | set2)

    def value_function(self, policy):
        V = np.zeros((len(self.states), 3))  # For engagement, diversity, and novelty
        for t in range(100):
            for s in self.states:
                action = policy[s]
                new_state = self.transition(s, action)
                reward = np.array([self.reward_functions[obj](action) for obj in self.reward_functions])
                V[s] += self.gamma ** t * reward
        return V

    def transition(self, state, action):
        return np.random.choice(self.states, p=[self.transition_probabilities[state][a] for a in self.actions])
