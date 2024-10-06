import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_violin(scores_dict, title):
    """
    Function to create a violin plot based on given scores dictionary.
    Arguments:
    - scores_dict: Dictionary with method names as keys and list of scores as values.
    """
    # Convert the scores_dict to a DataFrame for seaborn
    data = []
    for method, scores in scores_dict.items():
        for score in scores:
            data.append({'Method': method, 'Item Score': score})
    df = pd.DataFrame(data)

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Method', y='Item Score', data=df, inner='box', palette='Set2')
    plt.title(title)
    plt.xlabel('Method')
    plt.ylabel('Item Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'violin_plot_random_greedyoracle_{title[8:]}.png')  # Save the plot
    plt.show()



if __name__ == '__main__':
    # Assuming you have access to the experiment data
    # Example of more comprehensive data collection
    import numpy as np

    # Simulate a more realistic distribution of scores
    # Random method with scores between 0 and 1 for 100 runs
    random_scores = np.random.uniform(0.2, 0.6, 100)

    # Greedy Oracle method with scores between 0.8 and 1.0 for 100 runs
    greedy_oracle_scores = np.random.uniform(0.8, 1.0, 100)

    # Create a dictionary to store the data
    scores_dict = {
        'Random': random_scores,
        'Greedy Oracle': greedy_oracle_scores
    }

    # Plot the data
    plot_violin(scores_dict)
