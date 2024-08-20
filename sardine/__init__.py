'''
SARDINE
Copyright (c) 2023-present NAVER Corp. 
MIT license
'''
from gymnasium.envs.registration import register
from .version import __version__

num_items = [100, 200, 300, 500, 1000, 10000]
slate_sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20]

# for num_item in num_items:
#     for slate_size in slate_sizes:
#         register(
#             id = f"SlateTopK-BoredInf-v0-num_item{num_item}-slate_size{slate_size}",
#             entry_point = "sardine.simulator:Sardine",
#             kwargs = {
#                 "num_items": num_item,
#                 "slate_size": slate_size,
#                 "num_topics": 10,
#                 "episode_length": 100,
#                 "env_alpha": 1.0,
#                 "env_propensities": None,
#                 "env_offset": 0.65,
#                 "env_slope": 100,
#                 "env_omega": 0.95,
#                 "recent_items_maxlen": 10,
#                 "boredom_threshold": 5,
#                 "boredom_moving_window": 5,
#                 "env_embedds": None,
#                 "click_model": "tdPBM",
#                 "rel_penalty": False,
#                 "rel_threshold": None,
#                 "prop_threshold": None,
#                 "diversity_penalty": 1.0,
#                 "diversity_threshold": 4,
#                 "click_prop": 0.85,
#                 "boredom_type": "user_car",
#                 "boredom_reduce_rate": 0.0,
#             }
#         )


register(
    id = f"SlateTopK-BoredInf-v0-num_item1000-slate_size10",
    entry_point = "sardine.simulator:Sardine",
    kwargs = {
        "num_items": 1000,
        "slate_size": 10,
        "num_topics": 10,
        "episode_length": 100,
        "env_alpha": 1.0,
        "env_propensities": None,
        "env_offset": 0.65,
        "env_slope": 100,
        "env_omega": 0.95,
        "recent_items_maxlen": 10,
        "boredom_threshold": 5,
        "boredom_moving_window": 5,
        "env_embedds": None,
        "click_model": "tdPBM",
        "rel_penalty": False,
        "rel_threshold": None,
        "prop_threshold": None,
        "diversity_penalty": 1.0,
        "diversity_threshold": 4,
        "click_prop": 0.85,
        "boredom_type": "user_car",
        "boredom_reduce_rate": 0.0,
        "morl": True,
    }
)

register(
    id = f"SlateTopK-BoredInf-v0-num_item100-slate_size3",
    entry_point = "sardine.simulator:Sardine",
    kwargs = {
        "num_items": 100,
        "slate_size": 3,
        "num_topics": 10,
        "episode_length": 100,
        "env_alpha": 1.0,
        "env_propensities": None,
        "env_offset": 0.65,
        "env_slope": 100,
        "env_omega": 0.95,
        "recent_items_maxlen": 10,
        "boredom_threshold": 5,
        "boredom_moving_window": 5,
        "env_embedds": None,
        "click_model": "tdPBM",
        "rel_penalty": False,
        "rel_threshold": None,
        "prop_threshold": None,
        "diversity_penalty": 1.0,
        "diversity_threshold": 4,
        "click_prop": 0.85,
        "boredom_type": "user_car",
        "boredom_reduce_rate": 0.0,
        "morl": True,
    }
)


register(
    id="ml-100k-v0-morl",
    entry_point="sardine.simulator:Sardine",
    kwargs={
        "num_items": 1682,
        "slate_size": 8,
        "num_topics": 19,
        "episode_length": 100,
        "env_alpha": 1.0,
        "env_propensities": None,
        "env_offset": 0.65,
        "env_slope": 100,
        "env_omega": 1.0,
        "recent_items_maxlen": 10,
        "boredom_threshold": 5,
        "boredom_moving_window": 5,
        "env_embedds": "item_embeddings_ml-100k.npy",
        "click_model": "tdPBM",
        "rel_penalty": False,
        "rel_threshold": None,
        "prop_threshold": None,
        "diversity_penalty": 1.0,
        "diversity_threshold": 4,
        "click_prop": 0.85,
        "boredom_type": "user_car",
        "boredom_reduce_rate": 0.0,
        "user_priors": "user_priors_ml-100k.npz",
        "morl": True,
    }
)

register(
    id="ml-100k-v0",
    entry_point="sardine.simulator:Sardine",
    kwargs={
        "num_items": 1682,
        "slate_size": 8,
        "num_topics": 19,
        "episode_length": 100,
        "env_alpha": 1.0,
        "env_propensities": None,
        "env_offset": 0.65,
        "env_slope": 100,
        "env_omega": 1.0,
        "recent_items_maxlen": 10,
        "boredom_threshold": 5,
        "boredom_moving_window": 5,
        "env_embedds": "item_embeddings_ml-100k.npy",
        "click_model": "tdPBM",
        "rel_penalty": False,
        "rel_threshold": None,
        "prop_threshold": None,
        "diversity_penalty": 1.0,
        "diversity_threshold": 4,
        "click_prop": 0.85,
        "boredom_type": "user_car",
        "boredom_reduce_rate": 0.0,
        "user_priors": "user_priors_ml-100k.npz",
        "morl": False,
    }
)

register(
    id="sardine/SingleItem-Static-v0",
    entry_point="sardine.simulator:Sardine",
    kwargs={
        "num_items": 1000,
        "slate_size": 1,
        "num_topics": 10,
        "episode_length": 100,
        "env_alpha": 1.0,
        "env_propensities": None,
        "click_prop": 0.85,
        "env_offset": 0.65,
        "env_slope": 100,
        "env_omega": 1.0,
        "recent_items_maxlen": 10,
        "boredom_threshold": 11,
        "boredom_moving_window": 5,
        "env_embedds": "item_embeddings_focused_small.npy",
        "click_model": "tdPBM",
        "rel_penalty": False,
        "rel_threshold": None,
        "prop_threshold": None,
        "diversity_penalty": 1.0,
        "diversity_threshold": 4,
        "boredom_type": "user_car",
        "boredom_reduce_rate": 0.0,
    }
)

register(
    id="sardine/SingleItem-BoredInf-v0",
    entry_point="sardine.simulator:Sardine",
    kwargs={
        "num_items": 1000,
        "slate_size": 1,
        "num_topics": 10,
        "episode_length": 100,
        "env_alpha": 1.0,
        "env_propensities": None,
        "env_offset": 0.65,
        "env_slope": 100,
        "env_omega": 0.95,
        "recent_items_maxlen": 10,
        "boredom_threshold": 5,
        "boredom_moving_window": 5,
        "env_embedds": "item_embeddings_focused_small.npy",
        "click_model": "tdPBM",
        "rel_penalty": False,
        "rel_threshold": None,
        "prop_threshold": None,
        "diversity_penalty": 1.0,
        "diversity_threshold": 4,
        "click_prop": 0.85,
        "boredom_type": "user_car",
        "boredom_reduce_rate": 0.0,
    }
)

register(
    id="sardine/SingleItem-Uncertain-v0",
    entry_point="sardine.simulator:Sardine",
    kwargs={
        "num_items": 1000,
        "slate_size": 1,
        "num_topics": 10,
        "episode_length": 100,
        "env_alpha": 1.0,
        "env_propensities": None,
        "click_prop": 0.85,
        "env_offset": 0.65,
        "env_slope": 10, # 10 or 20, instead of 100
        "env_omega": 1.0,
        "recent_items_maxlen": 10,
        "boredom_threshold": 11,
        "boredom_moving_window": 5,
        "env_embedds": "item_embeddings_focused_small.npy",
        "click_model": "tdPBM",
        "rel_penalty": False,
        "rel_threshold": None,
        "prop_threshold": None,
        "diversity_penalty": 1.0,
        "diversity_threshold": 4,
        "boredom_type": "user_car",
        "boredom_reduce_rate": 0.0,
    }
)

register(
    id="sardine/SlateTopK-Bored-v0",
    entry_point="sardine.simulator:Sardine",
    kwargs={
        "num_items": 1000,
        "slate_size": 10,
        "num_topics": 10,
        "episode_length": 100,
        "env_alpha": 1.0,
        "env_propensities": None,
        "env_offset": 0.65,
        "env_slope": 100,
        "env_omega": 1.0,
        "recent_items_maxlen": 10,
        "boredom_threshold": 5,
        "boredom_moving_window": 5,
        "env_embedds": "item_embeddings_focused_small.npy",
        "click_model": "tdPBM",
        "rel_penalty": False,
        "rel_threshold": None,
        "prop_threshold": None,
        "diversity_penalty": 1.0,
        "diversity_threshold": 4,
        "click_prop": 0.85,
        "boredom_type": "user_car",
        "boredom_reduce_rate": 0.0,
    }
)

register(
    id="sardine/SlateTopK-BoredInf-v0",
    entry_point="sardine.simulator:Sardine",
    kwargs={
        "num_items": 1000,
        "slate_size": 10,
        "num_topics": 10,
        "episode_length": 100,
        "env_alpha": 1.0,
        "env_propensities": None,
        "env_offset": 0.65,
        "env_slope": 100,
        "env_omega": 0.95,
        "recent_items_maxlen": 10,
        "boredom_threshold": 5,
        "boredom_moving_window": 5,
        "env_embedds": "item_embeddings_focused_small.npy",
        "click_model": "tdPBM",
        "rel_penalty": False,
        "rel_threshold": None,
        "prop_threshold": None,
        "diversity_penalty": 1.0,
        "diversity_threshold": 4,
        "click_prop": 0.85,
        "boredom_type": "user_car",
        "boredom_reduce_rate": 0.0,
    }
)

register(
    id="sardine/SlateTopK-Uncertain-v0",
    entry_point="sardine.simulator:Sardine",
    kwargs={
        "num_items": 1000,
        "slate_size": 10,
        "num_topics": 10,
        "episode_length": 100,
        "env_alpha": 1.0,
        "env_propensities": None,
        "env_offset": 0.65,
        "env_slope": 10,
        "env_omega": 0.95,
        "recent_items_maxlen": 10,
        "boredom_threshold": 5,
        "boredom_moving_window": 5,
        "env_embedds": "item_embeddings_focused_small.npy",
        "click_model": "tdPBM",
        "rel_penalty": False,
        "rel_threshold": None,
        "prop_threshold": None,
        "diversity_penalty": 1.0,
        "diversity_threshold": 4,
        "click_prop": 0.85,
        "boredom_type": "user_car",
        "boredom_reduce_rate": 0.0,
    }
)

register(
    id="sardine/SlateRerank-Static-v0",
    entry_point="sardine.simulator:Sardine",
    kwargs={
        "num_items": 10,
        "slate_size": 10,
        "num_topics": 10,
        "episode_length": 10,
        "env_alpha": 1.0,
        "env_propensities": None,
        "env_offset": 0.3,
        "env_slope": 5,
        "env_omega": 1.0,
        "recent_items_maxlen": 10,
        "boredom_threshold": 11,
        "boredom_moving_window": 5,
        "env_embedds": "item_embeddings_rerank.npy",
        "click_model": "tdPBM",
        "rel_penalty": False,
        "rel_threshold": None,
        "prop_threshold": None,
        "diversity_penalty": 1.0,
        "diversity_threshold": 4,
        "click_prop": 0.7,
        "boredom_type": "user_car",
        "boredom_reduce_rate": 0.0,
    }
)

register(
    id="sardine/SlateRerank-Bored-v0",
    entry_point="sardine.simulator:Sardine",
    kwargs={
        "num_items": 10,
        "slate_size": 10,
        "num_topics": 10,
        "episode_length": 10,
        "env_alpha": 1.0,
        "env_propensities": None,
        "env_offset": 0.3,
        "env_slope": 5,
        "env_omega": 1.0,
        "recent_items_maxlen": 10,
        "boredom_threshold": 4,
        "boredom_moving_window": 5,
        "env_embedds": "item_embeddings_rerank.npy",
        "click_model": "tdPBM",
        "rel_penalty": False,
        "rel_threshold": None,
        "prop_threshold": None,
        "diversity_penalty": 1.0,
        "diversity_threshold": 4,
        "click_prop": 0.7,
        "boredom_type": "user_car",
        "boredom_reduce_rate": 0.0,
    }
)
