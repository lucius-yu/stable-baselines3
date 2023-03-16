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

def get_mask(observations: Union[np.ndarray, list]):
    apps = ['amf', 'cellf', 'ngf', 'uef', 'uplane']
    action_params = [ { "action": "None", "service_types": "None" },
            { "action": "up", "service_types": [ "amf" ] },
            { "action": "up", "service_types": [ "cellf" ] },
            { "action": "up", "service_types": [ "amf", "cellf" ] },
            { "action": "up", "service_types": [ "ngf" ] },
            { "action": "up", "service_types": [ "amf", "ngf" ] },
            { "action": "up", "service_types": [ "cellf", "ngf" ] },
            { "action": "up", "service_types": [ "uef" ] },
            { "action": "up", "service_types": [ "amf", "uef" ] },
            { "action": "up", "service_types": [ "cellf", "uef" ] },
            { "action": "up", "service_types": [ "ngf", "uef" ] },
            { "action": "up", "service_types": [ "uplane" ] },
            { "action": "up", "service_types": [ "amf", "uplane" ] },
            { "action": "up", "service_types": [ "cellf", "uplane" ] },
            { "action": "up", "service_types": [ "ngf", "uplane" ] },
            { "action": "up", "service_types": [ "uef", "uplane" ] },
            { "action": "down", "service_types": [ "amf" ] },
            { "action": "down", "service_types": [ "cellf" ] },
            { "action": "down", "service_types": [ "amf", "cellf" ] },
            { "action": "down", "service_types": [ "ngf" ] },
            { "action": "down", "service_types": [ "amf", "ngf" ] },
            { "action": "down", "service_types": [ "cellf", "ngf" ] },
            { "action": "down", "service_types": [ "uef" ] },
            { "action": "down", "service_types": [ "amf", "uef" ] },
            { "action": "down", "service_types": [ "cellf", "uef" ] },
            { "action": "down", "service_types": [ "ngf", "uef" ] },
            { "action": "down", "service_types": [ "uplane" ] },
            { "action": "down", "service_types": [ "amf", "uplane" ] },
            { "action": "down", "service_types": [ "cellf", "uplane" ] },
            { "action": "down", "service_types": [ "ngf", "uplane" ] },
            { "action": "down", "service_types": [ "uef", "uplane" ] }
    ]
    # get a single mask
    def get_single_mask(backends):
        result = []
        for act in action_params:
            act_res = True
            for svc in act['service_types']:
                if act['action'] == "up":
                    act_res = act_res and (backends[apps.index(svc)]<5)
                if act['action'] == "down":
                    act_res = act_res and (backends[apps.index(svc)]>1)
            result.append(act_res)
        return result
    # main function
    observations = np.array(observations) if type(observations) == list else observations
    if observations.ndim == 2:
        return np.array([get_single_mask(obs[8:13]) for obs in observations])
    if observations.ndim == 1:
        return np.array(get_single_mask(observations[8:13]))

# mask q values function
def mask_q_values(batch_observations: Union[np.ndarray, Dict[Union[str, int], np.ndarray]], batch_q_values: th.Tensor) -> th.Tensor:
    """
    Fetch mask information from obs, then mask out some q values and return

    :param obs:
    :param q_values
    :return: PyTorch tensor of the q_values, in it some q values has been replaced as mininum q value.
    """
    # return q values directly if env is not rlran-ifaas-v0
    if RLRAN_ENV_ID != 'rlran-ifaas-v0':
        print("Warning: mask_q_values only support gym env rlran-ifaas-v0")
        return batch_q_values

    if type(batch_observations) != np.ndarray:
        print("Warning: mask_q_values only support observation as type np.ndarray")
        return batch_q_values

    # mask q values with min of q values for actions which is not possible to execute
    batch_masks = get_mask(batch_observations) 
    masked_q_values = None
    for mask, q_values in zip(batch_masks, batch_q_values):
        mask = th.unsqueeze(th.tensor(mask),0)
        min_q_value = q_values.min()
        relative_q_values = q_values - min_q_value 
        relative_q_values = mask * relative_q_values
        masked_q_values = relative_q_values + min_q_value if masked_q_values == None else th.cat((masked_q_values, relative_q_values + min_q_value), dim=0)
    return masked_q_values


# check if the action is allowed according to backends information
def is_allowed_action(observation: Union[np.ndarray, Dict[Union[str, int], np.ndarray]], action: Union[np.ndarray,int]) -> bool:
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
        print(f"Warning: mask_q_values only support gym env rlran-ifaas-v0, not {RLRAN_ENV_ID}")
        return True

    if type(observation) != np.ndarray:
        print(f"Warning: mask_q_values only support observation as type np.ndarray, not {type(observation)}")
        return True

    observation = np.squeeze(observation)

    if observation.ndim > 1:
        print(f"Warning: observation dim > 1, return True. observation : {observation}")
        return True

    # get mask from observation
    mask = get_mask(observation)
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
