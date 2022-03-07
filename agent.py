from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import ray
#from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG

from ray.tune.logger import pretty_print
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models import ModelCatalog

from env import FixedMazeEnv

import logging
logging.basicConfig(level=logging.INFO)

# Start up Ray. This must be done before we instantiate any RL agents.
ray.init(num_cpus=1, num_gpus=1, ignore_reinit_error=True, log_to_driver=False)#, local_mode = True)

from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
#from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG

import models

config = DEFAULT_CONFIG.copy()
#See RLLIB doc for all of these parameters
#config['lambda']= 0.95
#config['model']['custom_model']= "convsimplemodel"
#config['kl_coeff'] = 0.0
#config['entropy_coeff'] = 0.01
#config['train_batch_size'] = 5000
#config['rollout_fragment_length'] = 5000
#config['sgd_minibatch_size']= 250
#config['num_sgd_iter']= 10
config['num_workers'] = 1
config['framework'] = "torch"
#config['lr'] = 3e-4
#config['num_cpus_per_worker']= 0
#config['num_gpus_per_worker']= 1
#config['batch_mode'] = 'truncate_episodes'
#config['observation_filter'] = 'NoFilter'
#config['model']['vf_share_layers'] = True
config['num_gpus']= 0
config['num_gpus_per_worker']= 0

config['model']['custom_model'] = 'rnn_model'
#config['model']['use_lstm'] = True
#config["replay_proportion"] = 0.5
#config["replay_buffer_num_slots"] = 200

#config['minibatch_buffer_size'] = 10
#config['num_sgd_iter'] = 3

#config['rollout_fragment_length'] = 50
#config['train_batch_size'] = 500

config['gamma'] = 0.99
#config['lr'] = 3e-3

#config["sgd_minibatch_size"] = 256
#config["train_batch_size"] = 400


#config['entropy_coeff'] = 0.01*5
"""config['exploration_config'] = {
    'type':'EpsilonGreedy',
    "initial_epsilon": 0.2,
    "final_epsilon": 0.02,
    "epsilon_timesteps": 100000}
"""

#Creates agent
agent = PPOTrainer(config, FixedMazeEnv)