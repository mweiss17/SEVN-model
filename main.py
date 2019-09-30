try:
    from comet_ml import Experiment
    comet_loaded = True
except ImportError:
    comet_loaded = False
import os
import time
from collections import deque

import numpy as np
import torch

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy, RandomPolicy, NaviBase
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate


def main():
    args = get_args()
    results_filename = f"logs/{args.env_name}-seed-{args.seed}-num-steps-{args.num_steps}-num-env-steps-{args.num_env_steps}-results.csv"
    save_path = os.path.join(args.save_dir, args.algo, str(args.seed))
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
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False,
                         args.custom_gym)

    if "Train" in args.env_name:
        test_envs = make_vec_envs(args.env_name.replace("Train", "Test"), args.seed, 1,
                             args.gamma, args.log_dir, device, False,
                             args.custom_gym)
    base = NaviBase
    obs_shape = envs.observation_space.shape

    try:
        os.makedirs(save_path)
        actor_critic, ob_rms = \
                    torch.load(save_path, map_location='cpu')
    except Exception:
        # create a new model
        actor_critic = Policy(
            obs_shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy},
            base=base,
        )

    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
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
        agent = algo.RANDOM_AGENT(actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

        actor_critic = RandomPolicy(obs_shape,
                                    envs.action_space,
                                    base_kwargs={'recurrent': args.recurrent_policy},
                                    base=base,
                                    )
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir,
            "trajs_{}.pt".format(args.env_name.split('-')[0].lower()))

        gail_train_loader = torch.utils.data.DataLoader(
            gail.ExpertDataset(
                file_name, num_trajectories=4, subsample_frequency=20),
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    test_rollouts = RolloutStorage(args.num_steps, 1,
                                   envs.observation_space.shape, envs.action_space,
                                   actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
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

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

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

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

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
