from _warnings import warn

from comet_ml import API
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(style="darkgrid")


api = API()
reported_metrics = ["Reward Mean", "Episodic Success Rate", "Episode Length Mean "]

def build_plot_dict(orig_env_name, raw_tuples, final_data, log_ts):
    for i in range(len(raw_tuples)):

        env_name = orig_env_name + ' - ' + reported_metrics[i]
        temp_data = np.array([list(x) for x in raw_tuples[i]]).transpose()

        # Preprocess for equal log intervals
        data = np.zeros((temp_data.shape[0], len(log_ts)))
        data[0, :] = log_ts
        for j in range(len(log_ts)):
            index = np.where(temp_data[0] == log_ts[j])[0]
            if index.size == 0:
                continue
            data[:, j] = temp_data[:, index[0]]

        # If same experiment with different seed
        if env_name in final_data:
            final_data[env_name]['n'] += 1
            data_concat = np.vstack((final_data[env_name]['data0'], data[1, :]))
            final_data[env_name]['data0'] = data_concat

        # If first experiment with these hyperparams.
        else:
            final_data[env_name] = {'metric': reported_metrics[i], 'data0': data, 'n': 1}

    return final_data


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


gps_exp_ids = {
    "SEVN-Test-NoisyGPS-1-v1-s0-10p": "mweiss17/navi-corl-2019/2d3670a8af1e4c9d83f06a889e02862e",
    "SEVN-Test-NoisyGPS-1-v1-s1-10p": "mweiss17/navi-corl-2019/604adedd9fdd4f0bb18fff5d76a4400f",
    "SEVN-Test-NoisyGPS-1-v1-s2-10p": "mweiss17/navi-corl-2019/76078c37a6934a24bb644bda65fcff0e",
    "SEVN-Test-NoisyGPS-5-v1-s0-10p": "mweiss17/navi-corl-2019/60fb0b41644c4ba89949eb03ada53948",
    "SEVN-Test-NoisyGPS-5-v1-s1-10p": "mweiss17/navi-corl-2019/a887d1de91c0463e94985441bbf8b126",
    "SEVN-Test-NoisyGPS-5-v1-s2-10p": "mweiss17/navi-corl-2019/76078c37a6934a24bb644bda65fcff0e",
    "SEVN-Test-NoisyGPS-25-v1-s0-10p": "mweiss17/navi-corl-2019/281db6264d9d46fb8b0ada97ecd19903",
    "SEVN-Test-NoisyGPS-25-v1-s1-10p": "mweiss17/navi-corl-2019/f4f8901bd4944340bd8beb2c203ccca9",
    "SEVN-Test-NoisyGPS-25-v1-s2-10p": "mweiss17/navi-corl-2019/69a88388d483456c9174453956414f97",
    "SEVN-Test-NoisyGPS-100-v1-s0-10p": "mweiss17/navi-corl-2019/5824c0711976465cb1c2363434246961",
    "SEVN-Test-NoisyGPS-100-v1-s1-10p": "mweiss17/navi-corl-2019/79e37ee59d0e41b094eafb5a70f8df23",
    "SEVN-Test-NoisyGPS-100-v1-s2-10p": "mweiss17/navi-corl-2019/cb839ab2d3444f3aaadfb8497c7604ad",
}

plot_info = {
    "SEVN-Test-NoisyGPS-1": {'color': '#404387', 'plot_name': 'NoisyGPS-1'},
    "SEVN-Test-NoisyGPS-5": {'color': '#22a784', 'plot_name': 'NoisyGPS-5'},
    "SEVN-Test-NoisyGPS-25": {'color': '#fde724', 'plot_name': 'NoisyGPS-25'},
    "SEVN-Test-NoisyGPS-100": {'color': '#29788e', 'plot_name': 'NoisyGPS-100'},
}

gps_exp_data = {}
min_numframes = 0000000
max_numframes = 100000000
for name, exp_id in gps_exp_ids.items():
    gps_exp_data[name] = {}
    experiment = api.get(exp_id)

    reward_mean = experiment.metrics_raw[reported_metrics[0]]
    reward_arr = np.array(reward_mean).transpose()
    reward_arr = reward_arr[:, np.where(np.logical_and(reward_arr[0] >= min_numframes,
                                                       reward_arr[0] < max_numframes))[0]]
    gps_exp_data[name][reported_metrics[0]] = {
        'mean': np.mean(reward_arr[1]),
        'std': np.std(reward_arr[1]),
        'raw_data': reward_arr
    }

    episodic_success_rate = experiment.metrics_raw[reported_metrics[1]]
    ep_succes_rate_arr = np.array(episodic_success_rate).transpose()
    ep_succes_rate_arr = ep_succes_rate_arr[:, np.where(np.logical_and(ep_succes_rate_arr[0] >= min_numframes,
                                                                       ep_succes_rate_arr[0] < max_numframes))[0]]
    gps_exp_data[name][reported_metrics[1]] = {
        'mean': np.mean(ep_succes_rate_arr[1]),
        'std': np.std(ep_succes_rate_arr[1]),
        'raw_data': ep_succes_rate_arr
    }

    episode_length_mean = experiment.metrics_raw[reported_metrics[2]]
    ep_length_mean_arr = np.array(episode_length_mean).transpose()
    ep_length_mean_arr = ep_length_mean_arr[:, np.where(np.logical_and(ep_length_mean_arr[0] >= min_numframes,
                                                                       ep_length_mean_arr[0] < max_numframes))[0]]
    gps_exp_data[name][reported_metrics[2]] = {
        'mean': np.mean(ep_length_mean_arr[1]),
        'std': np.std(ep_length_mean_arr[1]),
        'raw_data': ep_length_mean_arr
    }

from collections import defaultdict
exps = defaultdict(list)
for name, exp_id in gps_exp_ids.items():
    exp = "-".join(name.split("-")[:4])
    if exp in name:
        exps[exp].append(gps_exp_data[name])

compiled_data = []
for name, data in exps.items():
    c_data = {'name': name}
    for metric in reported_metrics:
        try:
            trials = np.concatenate([np.expand_dims(data[0][metric]['raw_data'][1], axis=0), np.expand_dims(data[1][metric]['raw_data'][1], axis=0), np.expand_dims(data[2][metric]['raw_data'][1], axis=0)], axis=0)
        except Exception as e:
            continue
        mean = trials.mean(axis=0)
        std = trials.std(axis=0)
        c_data[metric] = np.concatenate([np.expand_dims(data[0][metric]['raw_data'][0], axis=0), np.expand_dims(mean, axis=0), np.expand_dims(std, axis=0)], axis=0)
    compiled_data.append(c_data)

for metric in reported_metrics:
    # fig = plt.figure()
    # plt.title(metric, fontsize=18)
    # plt.xlabel('Timesteps', fontsize=14)
    # plt.ylabel(metric, fontsize=14)
    for data in compiled_data:
        color = plot_info[data['name']]['color']
        label = plot_info[data['name']]['plot_name']
        try:
            import pdb; pdb.set_trace()
            to_plot = pd.DataFrame(data[metric].T, columns=['timestep', 'mean', 'std'])
            sns.lineplot(x="Timesteps", y=metric, data=to_plot)
            # plt.errorbar(data0[metric][0], data0[metric][1], yerr=data0[metric][2], color=color, label=label, alpha=0.5)
        except Exception:
            pass
        # plt.plot(running_mean(data0[0], 100), running_mean(data0[1], 100), color, label=label)

    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # plt.legend(fontsize=14)
    # plt.savefig('plots/gps_exp'+metric + ".png")
    #
