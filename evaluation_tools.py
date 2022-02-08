import os
import numpy as np
import torch
import tools


def evaluate(PCT_policy, eval_envs, timeStr, args, eval_freq = 100, factor = 1):
    """Constructs the main actor & critic networks, and performs all training."""
    PCT_policy.eval()
    obs = eval_envs.reset()
    obs = torch.FloatTensor(obs).cuda().unsqueeze(dim=0)
    all_nodes, leaf_nodes = tools.get_leaf_nodes_with_factor(obs, 1 / factor, args.num_processes,
                                                             args.internal_node_holder, args.leaf_node_holder)
    batchX = torch.arange(args.num_processes)
    step_counter = 0
    episode_ratio = []
    episode_length = []
    all_episodes = []
    inner_counter = 0
    entropy_recorder = {}
    valid_leaf_node_recorder = {}
    prob_recorder = {}
    td_error_recorder = {}

    while step_counter < eval_freq:
        with torch.no_grad():
            selectedlogProb, selectedIdx, policy_dist_entropy, value = PCT_policy(all_nodes, True)
        selected_leaf_node = leaf_nodes[batchX, selectedIdx.squeeze()]
        items = eval_envs.packed
        obs, reward, done, infos = eval_envs.step(selected_leaf_node.cpu().numpy()[0][0:6] * factor)

        if inner_counter not in entropy_recorder.keys():
            entropy_recorder[inner_counter] = []
            valid_leaf_node_recorder[inner_counter] = []
            prob_recorder[inner_counter] = []

        if not done:
            entropy_recorder[inner_counter].append(policy_dist_entropy.item())
            valid_flag = all_nodes[:, args.internal_node_holder: args.internal_node_holder + args.leaf_node_holder, 8].cpu().numpy()
            valid_leaf_node_recorder[inner_counter].append(np.sum(valid_flag))
            prob_recorder[inner_counter].append(torch.exp(selectedlogProb).item())
            if inner_counter > 0:
                if inner_counter - 1  not in td_error_recorder.keys():
                    td_error_recorder[inner_counter - 1] = []
                error = abs(last_value.item() - last_reward - value.item())
                td_error_recorder[inner_counter - 1].append(error)

        last_value = value
        last_reward = reward

        inner_counter += 1

        if done:
            print('Episode {} ends.'.format(step_counter))
            if 'ratio' in infos.keys():
                episode_ratio.append(infos['ratio'])
            if 'counter' in infos.keys():
                episode_length.append(infos['counter'])

            print('Mean ratio: {}, length: {}'.format(np.mean(episode_ratio), np.mean(episode_length)))
            print('Episode ratio: {}, length: {}'.format(infos['ratio'], infos['counter']))
            all_episodes.append(items)
            step_counter += 1
            obs = eval_envs.reset()
            inner_counter = 0

        obs = torch.FloatTensor(obs).cuda().unsqueeze(dim=0)
        all_nodes, leaf_nodes = tools.get_leaf_nodes_with_factor(obs, 1 / factor, args.num_processes,
                                                                 args.internal_node_holder, args.leaf_node_holder)
        all_nodes, leaf_nodes = all_nodes.cuda(), leaf_nodes.cuda()

    result = "Evaluation using {} episodes\n" \
             "Mean ratio {:.5f}, mean length{:.5f}\n".format(len(episode_ratio), np.mean(episode_ratio), np.mean(episode_length))
    print(result)
    np.save(os.path.join('./logs/evaluation', timeStr, 'trajs.npy'), all_episodes)
    file = open(os.path.join('./logs/evaluation', timeStr, 'result.txt'), 'w')
    file.write(result)
    file.close()
