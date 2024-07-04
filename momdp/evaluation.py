import numpy as np

def evaluate_momdp(momdp, policy, item_popularity):
    long_term_engagement = sum([momdp.engagement_reward(policy[s]) for s in momdp.states])
    avg_intra_list_diversity = np.mean([momdp.diversity_reward(policy[s]) for s in momdp.states])
    avg_novelty = np.mean([momdp.novelty_reward(policy[s], item_popularity) for s in momdp.states])
    catalog_coverage = len(set([i for s in momdp.states for i in policy[s]])) / len(momdp.items)
    
    return {
        'Long-term Engagement': long_term_engagement,
        'Average Intra-list Diversity': avg_intra_list_diversity,
        'Average Novelty': avg_novelty,
        'Catalog Coverage': catalog_coverage
    }
