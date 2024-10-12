"""MORL algorithm base classes."""
import os
import time
from abc import ABC, abstractmethod
from distutils.util import strtobool
from typing import Dict, Optional, Union

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn
import wandb
from gymnasium import spaces
from mo_gymnasium.utils import MOSyncVectorEnv

from .evaluation import (
    eval_mo_reward_conditioned,
    policy_evaluation_mo,
)


class MOPolicy(ABC):
    """An MORL policy.

    It has an underlying learning structure which can be:
    - used to get a greedy action via eval()
    - updated using some experiences via update()

    Note that the learning structure can embed multiple policies (for example using a Conditioned Network).
    In this case, eval() requires a weight vector as input.
    """

    def __init__(self, id: Optional[int] = None, device: Union[th.device, str] = "auto") -> None:
        """Initializes the policy.

        Args:
            id: The id of the policy
            device: The device to use for the tensors
        """
        self.id = id
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu") if device == "auto" else device
        self.global_step = 0

    @abstractmethod
    def eval(self, obs: np.ndarray, w: Optional[np.ndarray]) -> Union[int, np.ndarray]:
        """Gives the best action for the given observation.

        Args:
            obs (np.array): Observation
            w (optional np.array): weight for scalarization

        Returns:
            np.array or int: Action
        """

    def __report(
        self,
        scalarized_return,
        scalarized_discounted_return,
        vec_return,
        discounted_vec_return,
        catalog_coverage
    ):
        """Writes the data to wandb summary."""
        if self.id is None:
            idstr = ""
        else:
            idstr = f"_{self.id}"

        wandb.log(
            {
                f"eval{idstr}/scalarized_return": scalarized_return,
                f"eval{idstr}/scalarized_discounted_return": scalarized_discounted_return,
                "global_step": self.global_step,
            }
        )
        for i in range(vec_return.shape[0]):
            n = 1
            if i == 0:
                # engagement
                name = "cumulative clicks"
            elif i == 1:
                name = "cumulative diversity"
                n = 100
            wandb.log(
                {f"eval{idstr}/{name}": vec_return[i]/n},
            )
        wandb.log(
            {f"eval{idstr}/catalog_coverage": catalog_coverage},
        )

    def policy_eval(
        self,
        eval_env,
        num_episodes: int = 5,
        scalarization=np.dot,
        weights: Optional[np.ndarray] = None,
        log: bool = False,
        state_encoder: th.nn.Module = None,
        num_envs: int = 1,
        ranker: str = "gems",
        observable: bool = False,
    ):
        """Runs a policy evaluation (typically over a few episodes) on eval_env and logs some metrics if asked.

        Args:
            eval_env: evaluation environment
            num_episodes: number of episodes to evaluate
            scalarization: scalarization function
            weights: weights to use in the evaluation
            log: whether to log the results

        Returns:
             a tuple containing the average evaluations
        """
        (
            scalarized_return,
            scalarized_discounted_return,
            vec_return,
            discounted_vec_return,
            catalog_coverage
        ) = policy_evaluation_mo(self, eval_env, scalarization=scalarization, w=weights, rep=num_episodes, state_encoder=state_encoder, num_envs = num_envs, ranker=ranker, observable=observable)

        if log:
            self.__report(
                scalarized_return,
                scalarized_discounted_return,
                vec_return,
                discounted_vec_return,
                catalog_coverage
            )

        return scalarized_return, scalarized_discounted_return, vec_return, discounted_vec_return, catalog_coverage

    def policy_eval_esr(
        self,
        eval_env,
        scalarization,
        weights: Optional[np.ndarray] = None,
        log: bool = False,
    ):
        """Runs a policy evaluation (typically on one episode) on eval_env and logs some metrics if asked.

        Args:
            eval_env: evaluation environment
            scalarization: scalarization function
            weights: weights to use in the evaluation
            log: whether to log the results

        Returns:
             a tuple containing the evaluations
        """
        (
            scalarized_reward,
            scalarized_discounted_reward,
            vec_reward,
            discounted_vec_reward,
        ) = eval_mo_reward_conditioned(self, eval_env, scalarization, weights)

        if log:
            self.__report(
                scalarized_reward,
                scalarized_discounted_reward,
                vec_reward,
                discounted_vec_reward,
            )

        return scalarized_reward, scalarized_discounted_reward, vec_reward, discounted_vec_reward

    def get_policy_net(self) -> torch.nn.Module:
        """Returns the weights of the policy net."""
        pass

    def get_buffer(self):
        """Returns a pointer to the replay buffer."""
        pass

    def set_buffer(self, buffer):
        """Sets the buffer to the passed buffer.

        Args:
            buffer: new buffer (potentially shared)
        """
        pass

    def set_weights(self, weights: np.ndarray):
        """Sets new weights.

        Args:
            weights: the new weights to use in scalarization.
        """
        pass

    @abstractmethod
    def update(self) -> None:
        """Update algorithm's parameters (e.g. using experiences from the buffer)."""


class MOAgent(ABC):
    """An MORL Agent, can contain one or multiple MOPolicies. Contains helpers to extract features from the environment, setup logging etc."""

    def __init__(self, env: Optional[gym.Env], device: Union[th.device, str] = "auto", seed: Optional[int] = None) -> None:
        """Initializes the agent.

        Args:
            env: (gym.Env): The environment
            device: (str): The device to use for training. Can be "auto", "cpu" or "cuda".
            seed: (int): The seed to use for the random number generator
        """
        self.extract_env_info(env)
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu") if device == "auto" else device

        self.global_step = 0
        self.num_episodes = 0
        self.seed = seed
        self.np_random = np.random.default_rng(self.seed)

    def extract_env_info(self, env: Optional[gym.Env]) -> None:
        """Extracts all the features of the environment: observation space, action space, ...

        Args:
            env (gym.Env): The environment
        """
        # Sometimes, the environment is not instantiated at the moment the MORL algorithms is being instantiated.
        # So env can be None. It is the responsibility of the implemented MORLAlgorithm to call this method in those cases
        if env is not None:
            self.env = env
            if isinstance(self.env.observation_space, spaces.Discrete):
                self.observation_shape = (1,)
                self.observation_dim = self.env.unwrapped.observation_space.n
            else:
                self.observation_shape = self.env.unwrapped.observation_space.shape
                # self.observation_dim = [i[1] for i in self.env.unwrapped.observation_space.items()]

            self.action_space = env.unwrapped.action_space
            if isinstance(self.env.unwrapped.action_space, (spaces.Discrete, spaces.MultiBinary)):
                self.action_shape = (1,)
                self.action_dim = self.env.unwrapped.action_space.n
            else:
                self.action_shape = self.env.unwrapped.action_space.shape
                self.action_dim = self.env.unwrapped.action_space.shape[0]
            # self.reward_dim = self.env.unwrapped.reward_space.shape[0]


    @abstractmethod
    def get_config(self) -> dict:
        """Generates dictionary of the algorithm parameters configuration.

        Returns:
            dict: Config
        """

    def register_additional_config(self, conf: Dict = {}) -> None:
        """Registers additional config parameters to wandb. For example when calling train().

        Args:
            conf: dictionary of additional config parameters
        """
        for key, value in conf.items():
            wandb.config[key] = value

    def setup_wandb(
        self, project_name: str, experiment_name: str, entity: Optional[str] = None, group: Optional[str] = None
    ) -> None:
        """Initializes the wandb writer.

        Args:
            project_name: name of the wandb project. Usually MORL-Baselines.
            experiment_name: name of the wandb experiment. Usually the algorithm name.
            entity: wandb entity. Usually your username but useful for reporting other places such as openrlbenmark.

        Returns:
            None
        """
        self.experiment_name = experiment_name
        env_id = self.env.spec.id if not isinstance(self.env, MOSyncVectorEnv) else self.env.envs[0].spec.id
        self.full_experiment_name = f"{env_id}__{experiment_name}__{self.seed}__{int(time.time())}"
        import wandb

        config = self.get_config()
        config["algo"] = self.experiment_name
        # looks for whether we're using a Gymnasium based env in env_variable
        monitor_gym = strtobool(os.environ.get("MONITOR_GYM", "True"))

        wandb.init(
            project=project_name,
            entity=entity,
            config=config,
            name=self.full_experiment_name,
            monitor_gym=monitor_gym,
            save_code=True,
            group=group,
        )
        # The default "step" of wandb is not the actual time step (gloabl_step) of the MDP
        wandb.define_metric("*", step_metric="global_step")

    def close_wandb(self) -> None:
        """Closes the wandb writer and finishes the run."""
        import wandb

        wandb.finish()
