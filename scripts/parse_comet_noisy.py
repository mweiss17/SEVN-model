from _warnings import warn

from comet_ml import API
import numpy as np
import matplotlib.pyplot as plt

api = API()

exp_ids = {
    "SEVN-Train-AllObs-Shaped-v1-s0-10p": "mweiss17/navi-corl-2019/008004e9c9a940e088437e4ddeab9eb4",
    "SEVN-Train-AllObs-Shaped-v1-s1-10p": "mweiss17/navi-corl-2019/dbd36b752d6a4703904161d95ee09868",
    "SEVN-Train-AllObs-Shaped-v1-s2-10p": "mweiss17/navi-corl-2019/84dbd53a36db4b39a7afc9acc66609a0",
    "SEVN-Train-AllObs-Shaped-v1-s3-10p": "mweiss17/navi-corl-2019/12f4aec90e284d1188bbe6307bdc33bd",
    "SEVN-Train-AllObs-Shaped-v1-s4-10p": "mweiss17/navi-corl-2019/bb6af29d7336411b92e31f750b5087bb",
}

# oracle_random_ids = {
#     "Hyrule-Mini-Random-v1": "mweiss17/navi-corl-2019/80b8b611c84242ffa61d08cc3364ba4b",
#     "Hyrule-Mini-Oracle-v1": "mweiss17/navi-corl-2019/c212813764de4a66994912dae21a8628",
# }

plot_info = {
    "SEVN-Train-AllObs-Shaped-v1": {'color': '#22a784', 'plot_name': 'AllObs'},
    # "Hyrule-Mini-NoImg-Shaped-v1": {'color': '#fde724', 'plot_name': 'NoImg'},
    # "Hyrule-Mini-NoGPS-Shaped-v1": {'color': '#440154', 'plot_name': 'NoGPS'},
    # "Hyrule-Mini-ImgOnly-Shaped-v1": {'color': '#29788e', 'plot_name': 'ImgOnly'},
    # "Hyrule-Mini-Random-v1": {'color': '#79d151', 'plot_name': 'Random'},
    # "Hyrule-Mini-Oracle-v1": {'color': '#404387', 'plot_name': 'Oracle'},
}

reported_metrics = ["Reward Mean", "Episodic Success Rate" , "Episode Length Mean ",]


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


# Training Data
final_data = {}
for name, exp_id in exp_ids.items():
    experiment = api.get(exp_id)
    for d in experiment.parameters:
        if d["name"] == 'env-name':
            assert d['valueMax'] in name
            env_name = d['valueMax']
        if d["name"] == 'log_interval':
            log_int = d['valueMax']

    reward_mean = experiment.metrics_raw[reported_metrics[0]]
    episodic_success_rate = experiment.metrics_raw[reported_metrics[1]]
    episode_length_mean = experiment.metrics_raw[reported_metrics[2]]
    if name == 'SEVN-Train-AllObs-Shaped-v1-s0-10p':
        logged_timesteps = np.array([list(x) for x in reward_mean]).transpose()[0]


    final_data = build_plot_dict(orig_env_name=env_name,
                                 raw_tuples=[reward_mean, episodic_success_rate, episode_length_mean],
                                 final_data=final_data,
                                 log_ts=logged_timesteps)

# Random-Oracle data0
# random_oracle_data = {}
# for name, exp_id in oracle_random_ids.items():
#     random_oracle_data[name] = {}
#     experiment = api.get(exp_id)
#
#     reward_mean = experiment.metrics_raw[reported_metrics[0]]
#     reward_arr = np.array(reward_mean).transpose()
#     random_oracle_data[name][reported_metrics[0]] = np.mean(reward_arr[1])
#
#     if "Oracle" in name:
#         random_oracle_data[name][reported_metrics[1]] = 1.0
#     else:
#         episodic_success_rate = experiment.metrics_raw[reported_metrics[1]]
#         ep_succes_rate_arr = np.array(episodic_success_rate).transpose()
#         random_oracle_data[name][reported_metrics[1]] = np.mean(ep_succes_rate_arr[1])
#
#     episode_length_mean = experiment.metrics_raw[reported_metrics[2]]
#     ep_length_mean_arr = np.array(episode_length_mean).transpose()
#     random_oracle_data[name][reported_metrics[2]] = np.mean(ep_length_mean_arr[1])


# Plotting Statistics
for metric in reported_metrics:
    fig = plt.figure()
    plt.title(metric, fontsize=18)
    plt.xlabel('Timesteps', fontsize=14)
    plt.ylabel(metric, fontsize=14)

    # Add random and oracle here.
    # for name, _ in oracle_random_ids.items():
    #     color = plot_info[name]['color']
    #     label = plot_info[name]['plot_name']
    #     plt.axhline(y=random_oracle_data[name][metric], color=color, label=label, linestyle='--')
    for key, val in final_data.items():
        if metric in key:
            key = key.replace(' - ' + metric, "")
            color = plot_info[key]['color']
            label = plot_info[key]['plot_name']

            met_mean = np.mean(val['data0'][1:], axis=0)
            met_std = np.std(val['data0'][1:], axis=0)

            plt.fill_between(running_mean(val['data0'][0], 10), running_mean(met_mean - met_std, 10),
                             running_mean(met_mean + met_std, 10), alpha=0.1, facecolor=color)
            plt.plot(running_mean(val['data0'][0], 10), running_mean(met_mean, 10), color, label=label)

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.legend(fontsize=14)
    plt.savefig('plots/'+metric + "123.png")


# Summary Statistics
for key, val in final_data.items():
    print(key)
    if val['metric'] == reported_metrics[0]:
        print("Reward Mean. Mean of the last 10 log intervals:", np.mean(np.mean(val['data0'][1:], axis=0)[-10:]))
    elif val['metric'] == reported_metrics[1]:
        print("Success Rate. Mean of max success rates:", np.mean(np.max(val['data0'][1:, -100:], axis=1)))
    elif val['metric'] == reported_metrics[2]:
        print("Episode Length Mean. Mean of the last 10 log intervals:", np.mean(np.mean(val['data0'][1:], axis=0)[-10:]))
    print("------------------------------------------------------------------------------------------------")


# GPS data0
# 4 procs only
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
    "SEVN-Train-AllObs-Shaped-v1": {'color': '#22a784', 'plot_name': 'AllObs'},
    "SEVN-Test-NoisyGPS-1-v1": {'color': '#fde724', 'plot_name': 'NoisyGPS-1'},
    "SEVN-Test-NoisyGPS-5-v1": {'color': '#fde724', 'plot_name': 'NoisyGPS-5'},
    "SEVN-Test-NoisyGPS-25-v1": {'color': '#fde724', 'plot_name': 'NoisyGPS-25'},
    "SEVN-Test-NoisyGPS-100-v1": {'color': '#fde724', 'plot_name': 'NoisyGPS-100'},
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

# Plotting Statistics
for metric in reported_metrics:
    fig = plt.figure()
    plt.title(metric, fontsize=18)
    plt.xlabel('Timesteps', fontsize=14)
    plt.ylabel(metric, fontsize=14)

    for key, val in gps_exp_data.items():
        key = "-".join(key.split('-')[:5])
        color = plot_info[key]['color']
        label = plot_info[key]['plot_name']

        data = val[metric]["raw_data"]
        plt.plot(running_mean(data[0], 100), running_mean(data[1], 100), color, label=label)

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.legend(fontsize=14)
    plt.savefig('plots/gps_exp'+metric + ".png")

