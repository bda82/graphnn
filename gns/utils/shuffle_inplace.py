import numpy as np


def shuffle_inplace(*args):
    """
    Shuffle data in place.

    Args:
        args: arguments
    
    Returns:
        shuffled arguments
    """
    rng_state = np.random.get_state()
    
    for a in args:
        np.random.set_state(rng_state)
        np.random.shuffle(a)
