try:
    from comet_ml import Experiment, ExistingExperiment
    comet_loaded = True
except ImportError:
    comet_loaded = False
import os
import json
import pickle
import time
from collections import deque

import numpy as np
import torch

from sevn_model import algo, utils
from sevn_model.arguments import get_args
from sevn_model.envs import make_vec_envs
from sevn_model.model import Policy, RandomPolicy, NaviBase, NaviBaseTemp
from sevn_model.storage import RolloutStorage
from evaluation import evaluate


def main():
    args = get_args()
    results_filename = f"logs/{args.env_name}-seed-{args.seed}-num-steps-{args.num_steps}-num-env-steps-{args.num_env_steps}-results.csv"
    save_path = os.path.join(args.save_dir, args.algo, str(args.seed))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.join(save_path, args.env_name)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, log_dir, device, False,
                         args.custom_gym)

    if "Train" in args.env_name:
        test_envs = make_vec_envs(args.env_name.replace("Train", "Test"), args.seed, 1,
                             args.gamma, log_dir, device, False,
                             args.custom_gym)
    base = NaviBaseTemp
    obs_shape = envs.observation_space.shape

    save_j = 0
    try:
        os.makedirs(save_path)
    except FileExistsError:
        pass

    # Recover from job pre-emption
    try:
        actor_critic, ob_rms = \
                    torch.load(os.path.join(save_path, args.env_name + ".pt"), map_location='cpu')
        j = json.load(open(os.path.join(save_path, args.env_name + "-state.json"), 'r'))
        save_j = j['save_j']
        episode_total = j['episode_total']
        test_episode_total = j['test_episode_total']

        rollouts = pickle.load(open(os.path.join(save_path, args.env_name + "-rollout.pkl"), 'rb'))
        rollouts.to(device)
        obs = envs.reset()
        rollouts.obs[0].copy_(obs)
        rollouts.to(device)

        test_rollouts = pickle.load(open(os.path.join(save_path, args.env_name + "-test-rollout.pkl"), 'rb'))
        test_rollouts.to(device)
        test_obs = test_envs.reset()
        test_rollouts.obs[0].copy_(test_obs)
        test_rollouts.to(device)

        optimizer_state_dict = pickle.load(open(os.path.join(save_path, args.env_name + "-optim-state-dict.pkl"), 'rb'))
        episode_rewards = pickle.load(open(os.path.join(save_path, args.env_name + "-episode_rewards.pkl"), 'rb'))
        episode_length = pickle.load(open(os.path.join(save_path, args.env_name + "-episode_length.pkl"), 'rb'))
        episode_success_rate = pickle.load(open(os.path.join(save_path, args.env_name + "-episode_success_rate.pkl"), 'rb'))
        test_episode_rewards = pickle.load(open(os.path.join(save_path, args.env_name + "-test_episode_rewards.pkl"), 'rb'))
        test_episode_length = pickle.load(open(os.path.join(save_path, args.env_name + "-test_episode_length.pkl"), 'rb'))
        test_episode_success_rate = pickle.load(open(os.path.join(save_path, args.env_name + "-test_episode_success_rate.pkl"), 'rb'))

        if comet_loaded and len(args.comet) > 0:
            comet_credentials = args.comet.split("/")
            experiment = ExistingExperiment(
                api_key=comet_credentials[2],
                previous_experiment=j['comet_id'])
            for key, value in vars(args).items():
                experiment.log_parameter(key, value)
        else:
            experiment = None
            with open(results_filename, "a") as f:
                for key, value in vars(args).items():
                    f.write(f"{key}, {value}\n")
                f.close()

    except Exception:
        # create a new model
        actor_critic = Policy(
            obs_shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy},
            base=base,
        )
        rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                  envs.observation_space.shape, envs.action_space,
                                  actor_critic.recurrent_hidden_state_size)
        import pdb; pdb.set_trace()
        obs = envs.reset()
        rollouts.obs[0].copy_(obs)
        rollouts.to(device)

        test_rollouts = RolloutStorage(args.num_steps, 1,
                                       envs.observation_space.shape, envs.action_space,
                                       actor_critic.recurrent_hidden_state_size)
        if "Train" in args.env_name:
            test_obs = test_envs.reset()
            test_rollouts.obs[0].copy_(test_obs)
            test_rollouts.to(device)

        episode_rewards = deque(maxlen=10)
        episode_length = deque(maxlen=10)
        episode_success_rate = deque(maxlen=100)
        episode_total = 0

        test_episode_rewards = deque(maxlen=10)
        test_episode_length = deque(maxlen=10)
        test_episode_success_rate = deque(maxlen=100)
        test_episode_total = 0

        if comet_loaded and len(args.comet) > 0:
            comet_credentials = args.comet.split("/")
            experiment = Experiment(
                api_key=comet_credentials[2],
                project_name=comet_credentials[1],
                workspace=comet_credentials[0])
            for key, value in vars(args).items():
                experiment.log_parameter(key, value)
        else:
            experiment = None
            with open(results_filename, "w+") as f:
                for key, value in vars(args).items():
                    f.write(f"{key}, {value}\n")
                f.close()

    actor_critic.to(device)

    if args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'random':
        agent = algo.RANDOM_AGENT(actor_critic, args.value_loss_coef, args.entropy_coef)

        actor_critic = RandomPolicy(obs_shape,
                                    envs.action_space,
                                    base_kwargs={'recurrent': args.recurrent_policy},
                                    base=base,
                                    )
    try:
        agent.optimizer.load_state_dict(optimizer_state_dict)
    except Exception:
        pass

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates - save_j):
        j = j + save_j
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates, args.lr)

        print("args.num_steps: " + str(args.num_steps))
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Observe reward and next obs
            obs, reward, done, infos = envs.step(action)
            for idx, info in enumerate(infos):
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    episode_length.append(info['episode']['l'])
                    if "Explorer" not in args.env_name:
                        episode_success_rate.append(info['was_successful_trajectory'])
                    episode_total += 1

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # Run on test
        if "Train" in args.env_name:
            for step in range(args.num_steps):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        test_rollouts.obs[step], test_rollouts.recurrent_hidden_states[step],
                        test_rollouts.masks[step])

                    # Observe reward and next obs
                    obs, reward, done, infos = test_envs.step(action)
                    for idx, info in enumerate(infos):
                        if 'episode' in info.keys():
                            test_episode_rewards.append(info['episode']['r'])
                            test_episode_length.append(info['episode']['l'])
                            test_episode_success_rate.append(info['was_successful_trajectory'])
                            test_episode_total += 1

        # Increment curriculum
        if np.mean(test_episode_success_rate) > 0.5:
            level = int(args.env_name.split("-")[-2]) + 1
            split = args.env_name.split("-")
            split[-2] = str(level)
            env_name = "-".join(split)
            args.env_name = env_name
            envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                                 args.gamma, log_dir, device, False,
                                 args.custom_gym)
            test_envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                                 args.gamma, log_dir, device, False,
                                 args.custom_gym)
            obs = envs.reset()
            rollouts.obs[0].copy_(obs)
            rollouts.to(device)
            test_obs = test_envs.reset()
            test_rollouts.obs[0].copy_(test_obs)
            test_rollouts.to(device)

            print(f"graduation to level: {level}")

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or
                j == num_updates - 1) and args.save_dir != "" and j > args.save_after:
            if args.save_multiple:
                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ], os.path.join(save_path, str(j) + "-" + args.env_name + ".pt"))
            else:
                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ], os.path.join(save_path, args.env_name + ".pt"))
                json.dump({'save_j': j, 'episode_total': episode_total, 'test_episode_total': test_episode_total, 'comet_id': experiment.id}, open(os.path.join(save_path, args.env_name + "-state.json"), 'w+'))
                pickle.dump(agent.optimizer.state_dict(),  open(os.path.join(save_path, args.env_name + "-optim-state-dict.pkl"), 'wb+'))
                pickle.dump(rollouts, open(os.path.join(save_path, args.env_name + "-rollout.pkl"), 'wb+'))
                pickle.dump(test_rollouts, open(os.path.join(save_path, args.env_name + "-test-rollout.pkl"), 'wb+'))
                pickle.dump(episode_rewards, open(os.path.join(save_path, args.env_name + "-episode_rewards.pkl"), 'wb+'))
                pickle.dump(episode_length, open(os.path.join(save_path, args.env_name + "-episode_length.pkl"), 'wb+'))
                pickle.dump(episode_success_rate, open(os.path.join(save_path, args.env_name + "-episode_success_rate.pkl"), 'wb+'))
                pickle.dump(test_episode_rewards, open(os.path.join(save_path, args.env_name + "-test_episode_rewards.pkl"), 'wb+'))
                pickle.dump(test_episode_length, open(os.path.join(save_path, args.env_name + "-test_episode_length.pkl"), 'wb+'))
                pickle.dump(test_episode_success_rate, open(os.path.join(save_path, args.env_name + "-test_episode_success_rate.pkl"), 'wb+'))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            if experiment is not None:
                experiment.log_metric("Reward Mean", np.mean(episode_rewards), step=total_num_steps)
                experiment.log_metric("Reward Min", np.min(episode_rewards), step=total_num_steps)
                experiment.log_metric("Reward Max", np.max(episode_rewards), step=total_num_steps)
                experiment.log_metric("Episode Length Mean ", np.mean(episode_length), step=total_num_steps)
                experiment.log_metric("Episode Length Min", np.min(episode_length), step=total_num_steps)
                experiment.log_metric("Episode Length Max", np.max(episode_length), step=total_num_steps)
                experiment.log_metric("# Trajectories (Total)", j, step=total_num_steps)
                if "Explorer" not in args.env_name:
                    experiment.log_metric("Episodic Success Rate", np.mean(episode_success_rate), step=total_num_steps)
            else:
                with open(results_filename, "a") as f:
                    f.write(f"Reward Mean, {np.mean(episode_rewards)}, {total_num_steps}\n")
                    f.write(f"Reward Min, {np.min(episode_rewards)}, {total_num_steps}\n")
                    f.write(f"Reward Max, {np.max(episode_rewards)}, {total_num_steps}\n")
                    f.write(f"Episode Length Mean, {np.mean(episode_rewards)}, {total_num_steps}\n")
                    f.write(f"Episode Length Min, {np.min(episode_rewards)}, {total_num_steps}\n")
                    f.write(f"Episode Length Max, {np.max(episode_rewards)}, {total_num_steps}\n")
                    f.write(f"# Trajectories (Total), {j}, {total_num_steps}\n")
                    if "Explorer" not in args.env_name:
                        f.write(f"Episodic Success Rate, {np.mean(episode_success_rate)}, {total_num_steps}\n")
                    f.close()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

            # Test Generalization
            if "Train" in args.env_name and j % args.log_interval == 0 and len(test_episode_rewards) > 1:
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])
                test_rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)

                with torch.no_grad():
                    next_value = actor_critic.get_value(
                        test_rollouts.obs[-1], test_rollouts.recurrent_hidden_states[-1],
                        test_rollouts.masks[-1]).detach()
                test_rollouts.after_update()

                print(f"Test Episode Total: {test_episode_total}, Mean Test rewards: {np.mean(test_episode_rewards)}, Test Episode Length: {np.mean(test_episode_length)}, Test Episode Success Rate: {np.mean(test_episode_success_rate)}")
                test_total_num_steps = (j + 1) * args.num_steps
                experiment.log_metric("Test Reward Mean", np.mean(test_episode_rewards), step=test_total_num_steps)
                experiment.log_metric("Test Reward Min", np.min(test_episode_rewards), step=test_total_num_steps)
                experiment.log_metric("Test Reward Max", np.max(test_episode_rewards), step=test_total_num_steps)
                experiment.log_metric("Test Episode Length Mean ", np.mean(test_episode_length), step=test_total_num_steps)
                experiment.log_metric("Test Episode Length Min", np.min(test_episode_length), step=test_total_num_steps)
                experiment.log_metric("Test Episode Length Max", np.max(test_episode_length), step=test_total_num_steps)
                experiment.log_metric("# Test Trajectories (Total)", j)
                experiment.log_metric("Test Episodic Success Rate", np.mean(test_episode_success_rate), step=test_total_num_steps)


        if (args.eval_interval is not None and len(episode_rewards) > 1 and
                j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
