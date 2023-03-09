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

# lambda function to extract backends from observation, it is correlated to your state space design
get_backends=lambda obs: obs[:,8:13] if obs.ndim==2 else obs[8:13]

# lambda function to calculate mask. if backends of service>1 then it could scaled down. if backends of service < 5 then it could be scaled up. it is correlated to your action mappsing
get_mask=lambda obs: np.apply_along_axis(lambda backends: np.array([True]+[True if v<5 else False for v in backends]+[True if v>1 else False for v in backends]), axis=obs.ndim-1,arr=obs)


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
    for backends, q_values in zip(get_backends(batch_observations), batch_q_values):
        mask = th.unsqueeze(th.tensor(get_mask(backends)),0)
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

    # calculate True False mask
    backends = get_backends(observation).squeeze()
    mask = get_mask(backends)
    
    # return masked action
    if type(action) == np.ndarray or type(action) == list:
        return all([mask[act] for act in action]) if len(action) > 0 else False
    else:
        return mask[action]

def get_ewma(beta=0.9, auto_correction=True):
    t = 0  # time steps
    A = 0  # ewma value
    beta = beta
    correction = auto_correction
    def ewma(v):
        nonlocal t, A, correction
        t += 1
        A = beta*A + (1-beta)*v
        if correction: 
            correction = beta**t > 0.001
            return A/(1-beta**t)
        else:
            return A
    return ewma
