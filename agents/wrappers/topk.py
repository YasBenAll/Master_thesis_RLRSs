"""TopK ranking with ideal item embeddings."""
from typing import Union

import numpy as np
import torch
import gymnasium as gym


class TopK(gym.ActionWrapper):
    def __init__(
        self,
        env: gym.Env,
        embeddings: str = "ideal",
        min_action: Union[float, np.ndarray] = None,
        max_action: Union[float, np.ndarray] = None,
    ):
        assert np.less_equal(min_action, max_action).all(), (min_action, max_action)
        super().__init__(env)

        self.env = env
        self.slate_size = env.unwrapped.slate_size
        self.embedd_dim = env.num_topics

        if embeddings == "ideal":
            self.embeddings = self.env.item_embedd
        else:
            self.embeddings = torch.load(embeddings)
            # convert to numpy
            self.embeddings = self.embeddings.detach().cpu().numpy()

        if min_action is None:
            min_action = np.min(self.embeddings)
        if max_action is None:
            max_action = np.max(self.embeddings)
        self.action_space = gym.spaces.Box(low = np.zeros(self.embedd_dim, dtype=np.float32) + min_action,
                                        high = np.zeros(self.embedd_dim, dtype=np.float32) + max_action,
                                        shape=(self.embedd_dim,), dtype=np.float32)

    def action(self, action):
        dot_product = self.embeddings @ action.squeeze()
        ind = np.argpartition(dot_product, - self.slate_size)[- self.slate_size:]
        slate = ind[np.flip(np.argsort(dot_product[ind]))]
        self.latest_slate = slate
        return slate