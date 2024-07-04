import numpy as np

def create_policy(momdp):
    return {s: np.random.choice(momdp.actions) for s in momdp.states}
