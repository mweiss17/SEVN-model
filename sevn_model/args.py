import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='ppo', help='algorithm to use: ppo | random')
    parser.add_argument(
        '--save-dir',
        default='~data/trained_models/sevn-model/',
        help='directory to save agent logs (default: ~data/trained_models/sevn-model/)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")

    parser.add_argument(
        '--log-dir',
        default='/tmp/gym/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--custom-gym',
        default='',
        help='import some dependency package for thew gym env')
    parser.add_argument(
        '--comet',
        default='',
        help='add comet.ml credentials in the format workspace/project/api_key')
    parser.add_argument(
        '--continue-model',
        help='continue training from model weights')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
