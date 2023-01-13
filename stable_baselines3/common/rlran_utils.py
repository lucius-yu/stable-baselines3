import glob
import os
import platform
import random
from collections import deque
from itertools import zip_longest
from typing import Dict, Iterable, List, Optional, Tuple, Union

import gym
import numpy as np
import torch as th

import stable_baselines3 as sb3

from stable_baselines3.common.logger import Logger, configure
from stable_baselines3.common.type_aliases import GymEnv, Schedule, TensorDict, TrainFreq, TrainFrequencyUnit


# mask q value only when RLRAN_ENV_ID == 'rlran-ifaas-v0'
RLRAN_ENV_ID = os.environ.get("RLRAN_ENV_ID", "")

# mask q values function
def mask_q_values(
    batch_observations: Union[np.ndarray, Dict[Union[str, int], np.ndarray]], batch_q_values: th.Tensor
) -> th.Tensor:
    """
    Fetch mask information from obs, then mask out some q values and return

    :param obs:
    :param q_values
    :return: PyTorch tensor of the q_values, in it some q values has been replaced as mininum q value.
    """

    # return q values directly if env is not rlran-ifaas-v0
    if RLRAN_ENV_ID != 'rlran-ifaas-v0':
        return batch_q_values

    # mask q values with min of q values for actions which is not possible to execute
    masked_q_values = None
    for backends, q_values in zip(batch_observations['backends'], batch_q_values):
        mask = th.tensor([[True, True]]+[[False,True] if v == 0 else [True, False] if v == 4 else [True, True] for v in backends]).flatten()[1:]
        mask = th.unsqueeze(mask, 0)
        min_q_value = q_values.min()
        relative_q_values = q_values - min_q_value 
        relative_q_values = mask * relative_q_values
        masked_q_values = relative_q_values + min_q_value if masked_q_values == None else th.cat((masked_q_values, relative_q_values + min_q_value), dim=0)

    return masked_q_values


# check if the action is allowed according to backends information
def is_allowed_action(
    observation: Union[np.ndarray, Dict[Union[str, int], np.ndarray]], action: Union[np.ndarray,int]
) -> bool:
    """
    Fetch mask information from obs, then mask out some q values and return

    :param obs:
    :param q_values
    :return: PyTorch tensor of the q_values, in it some q values has been replaced as mininum q value.
    """

    # action is None
    if action == None:
        return False
    # return q values directly if env is not rlran-ifaas-v0
    if RLRAN_ENV_ID != 'rlran-ifaas-v0':
        print(f'RLRAN_ENV_ID={RLRAN_ENV_ID}')
        return True
    if type(action) == np.ndarray:
        print(f'type(action)={type(action)}')
        return True    
    # calculate True False mask
    backends = observation['backends'].squeeze()
    mask=np.array([[True, True]]+[[False,True] if v == 0 else [True, False] if v == 4 else [True, True] for v in backends]).flatten()[1:]

    return mask[action]
