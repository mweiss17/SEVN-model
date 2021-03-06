import argparse
import os
# workaround to unpickle olf model files
import sys
import time

import numpy as np
import torch

from sevn_model.envs import VecPyTorch, make_vec_envs
from sevn_model.utils import get_render_func, get_vec_normalize

sys.path.append('sevn_model')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='log interval, one log per n updates (default: 10)')
parser.add_argument(
    '--env-name',
    default='PongNoFrameskip-v4',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--load-dir',
    default=None,
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument(
    '--load-model',
    default='./trained_models/ppo/0/SEVN-Mini-All-Shaped-v1.pt',
    help='a path to a particular model')
parser.add_argument(
    '--custom-gym',
    default='SEVN_gym',
    help='The gym to load from')
parser.add_argument(
    '--non-det',
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')
args = parser.parse_args()

args.det = not args.non_det

env = make_vec_envs(
    args.env_name,
    args.seed + 1000,
    1,
    None,
    None,
    device='cpu',
    custom_gym=args.custom_gym,
    allow_early_resets=False)

# Get a render function
render_func = get_render_func(env)

# We need to use the same statistics for normalization as used in training
if args.load_dir is not None:
    actor_critic, ob_rms = \
                torch.load(os.path.join(args.load_dir, args.env_name + ".pt"), map_location='cpu')
else:
    actor_critic, ob_rms = \
                torch.load(args.load_model, map_location='cpu')

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1,
                                      actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

obs = env.reset()
render_func('rgb_array', clear=True, first_time=True)

while True:
    i = 0
    done = False
    start = time.time()
    r = 0

    while i < 300 and not done:
        print(i)
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=args.det)
        # Obser reward and next obs
        obs, reward, done, _ = env.step(action)
        r += reward
        print("reward: " + str(r))
        print(f"action: {action}, reward: {reward}, done: {done}")
        masks.fill_(0.0 if done else 1.0)
        print("acted: " + str(time.time() - start))
        start = time.time()
        if render_func is not None:
            render_func('rgb_array', clear=False, first_time=False)
        print("rendered: " + str(time.time() - start))

        i += 1
    render_func('rgb_array', clear=True, first_time=False)
