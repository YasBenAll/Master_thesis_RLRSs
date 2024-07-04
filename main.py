import gymnasium as gym
from momdp.momdp import MOMDP
from momdp.policy import create_policy
from momdp.evaluation import evaluate_momdp
import numpy as np
import sardine
from sardine.wrappers import IdealState
from sardine.policies import EpsilonGreedyOracle

np.set_printoptions(precision=3)



def main():

    env = gym.make("sardine/test-v0")

    logging_policy = EpsilonGreedyOracle(epsilon = 0.0, env = env, seed = 2023)
    dataset = env.generate_dataset(n_users = 10, policy = logging_policy, seed = 2023, dataset_type="dict")

    ## If you want to work with Fully observable state, add a wrapper to the environment
    env = IdealState(env)

    observation, info = env.reset(seed = 2024)
    env.action_space.seed(2024)
    cum_reward_list, cum_boredom_list = [], []

    cum_reward = np.zeros(3)  # For engagement, diversity, and novelty
    cum_boredom = 0

    ep = 0

    while ep < 100:
        print(ep)
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        clicks = info["clicks"]
        slate = info["slate"]
        rewards = np.array([reward["engagement"], reward["diversity"], reward["novelty"]])
        cum_reward += rewards
        cum_boredom += (1.0 if np.sum(info["bored"] == True) > 0 else 0.0)

        if terminated or truncated:
            cum_reward_list.append(cum_reward)
            cum_boredom_list.append(cum_boredom)
            cum_reward = 0
            cum_boredom = 0
            observation, info = env.reset()
            ep += 1
    print(cum_reward_list)
    print("Average return: ", np.mean(cum_reward_list, axis=0))


if __name__ == "__main__":
    main()
