import os
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt

rewards = defaultdict(list)
successes = defaultdict(list)
trajectories = defaultdict(list)
data_dir = "results/"
exps = ["allobs", "imgonly", "nogps", "noimg"]
for exp in exps:
    exp_dir = data_dir + exp
    for f in os.listdir(exp_dir):
        results = np.load(os.path.join(exp_dir, f))
        exp_type = '-'.join(f.split('-')[:4])
        # if "SEVN-Test-AllObs-Shaped-v1" in f and ("s31" in f or "s36" in f):
        #     continue
        print(f'{f}: {results["successes"].mean()}')
        rewards[exp_type].append(results['rewards'].mean())
        successes[exp_type].append(results['successes'].mean())
        trajectories[exp_type].append(results['trajectories'].mean())

fig = plt.figure()
plt.title("Success Rate by Available Sensors")
plt.boxplot(x=successes.values(), whis=2)
plt.ylabel("Success Rate")
plt.xticks([1, 2, 3, 4], labels=["AllObs", "ImgOnly", "NoGPS", "NoImg"], rotation=0)
plt.savefig("plots/success_rate_boxplot.svg", format="svg")

fig = plt.figure()
plt.title("Rewards by Available Sensors")
plt.boxplot(x=rewards.values(), whis=2)
plt.ylabel("Trajectory Reward")
plt.xticks([1, 2, 3, 4], labels=["AllObs", "ImgOnly", "NoGPS", "NoImg"], rotation=0)
plt.savefig("plots/rewards_boxplot.svg", format="svg")

fig = plt.figure()
plt.title("Trajectories by Available Sensors")
plt.boxplot(x=trajectories.values(), whis=2)
plt.ylabel("Trajectory Length")
plt.xticks([1, 2, 3, 4], labels=["AllObs", "ImgOnly", "NoGPS", "NoImg"], rotation=0)
plt.savefig("plots/trajectory_boxplot.svg", format="svg")

# for exp_name, vals in rewards.items():
#     print(f'mean of rewards for {exp_name}: {np.array(rewards[exp_name]).mean():.3f}')
#     print(f'std  of rewards for {exp_name}: {np.array(rewards[exp_name]).std():.3f}')
# for exp_name, vals in rewards.items():
#     print(f'mean of successes for {exp_name}: {np.array(successes[exp_name]).mean():.3f}')
#     print(f'std of successes for {exp_name}: {np.array(successes[exp_name]).std():.3f}')
# for exp_name, vals in rewards.items():
#     print(f'mean of trajectory_length for {exp_name}: {np.array(trajectories[exp_name]).mean():.3f}')
#     print(f'std  of trajectory_length for {exp_name}: {np.array(trajectories[exp_name]).std():.3f}')
