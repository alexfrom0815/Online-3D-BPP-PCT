import os
import torch.optim as optim
import numpy as np
import tools
import time
from collections import deque
from storage import PCTRolloutStorage
from kfac import KFACOptimizer
import random
import torch
np.set_printoptions(threshold=np.inf)

class train_tools(object):
    def __init__(self, writer, timeStr, PCT_policy, args):
        self.writer = writer
        self.timeStr = timeStr
        self.step_counter = 0
        self.PCT_policy = PCT_policy
        self.use_acktr = args.use_acktr
        seed = args.seed

        if self.use_acktr:
            self.policy_optim = KFACOptimizer(self.PCT_policy) # For ACKTR method. （https://proceedings.neurips.cc/paper/2017/hash/361440528766bbaaaa1901845cf4152b-Abstract.html）
        else:
            self.policy_optim = optim.Adam(self.PCT_policy.parameters(), lr=args.learning_rate) # For naive A2C method.

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

    def train_n_steps(self, envs, args, device):
        model_save_path = os.path.join(args.model_save_path, self.timeStr)
        sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
        self.PCT_policy.train()
        factor = args.normFactor # NormFactor controlls the maximum value of the original input of the network to less than 1.0, which helps the training of the network

        obs = envs.reset()
        all_nodes, leaf_nodes = tools.get_leaf_nodes(obs, args.internal_node_holder, args.leaf_node_holder)
        all_nodes, leaf_nodes = all_nodes.to(device), leaf_nodes.to(device)
        pct_rollout = PCTRolloutStorage(args.num_steps,
                                        args.num_processes,
                                        obs_shape=all_nodes.shape[1:],
                                        gamma = args.gamma)
        pct_rollout.to(device)

        start = time.time()
        ratio_recorder = 0
        episode_rewards = deque(maxlen=10)
        episode_ratio = deque(maxlen=10)
        batchX = torch.arange(args.num_processes)

        inside_counter = self.step_counter
        num_steps, num_processes = args.num_steps, args.num_processes
        pct_rollout.obs[0].copy_(all_nodes)

        while True:
            ##############################################
            ####### Collect n-step training sample #######
            ##############################################
            self.step_counter += 1
            for step in range(num_steps):
                with torch.no_grad():
                    selectedlogProb, selectedIdx, dist_entropy, _ = self.PCT_policy(all_nodes, normFactor = factor)
                selected_leaf_node = leaf_nodes[batchX,selectedIdx.squeeze()]
                obs, reward, done, infos = envs.step(selected_leaf_node.cpu().numpy())
                all_nodes, leaf_nodes = tools.get_leaf_nodes(obs, args.internal_node_holder, args.leaf_node_holder)
                all_nodes, leaf_nodes = all_nodes.to(device), leaf_nodes.to(device)
                pct_rollout.insert(all_nodes, selectedIdx, selectedlogProb, reward, torch.tensor(1-done).unsqueeze(1))

            for _ in range(len(infos)):
                if done[_]:
                    if 'reward' in infos[_].keys():
                        episode_rewards.append(infos[_]['reward'])
                    else:
                        episode_rewards.append(infos[_]['episode']['r'])
                    if 'ratio' in infos[_].keys():
                        episode_ratio.append(infos[_]['ratio'])

            with torch.no_grad():
                _, _, _, next_value = self.PCT_policy(pct_rollout.obs[-1], normFactor = factor)

            pct_rollout.compute_returns(next_value)

            ##############################################
            ########### PCT policy optimzation ###########
            ##############################################
            obs_shape = pct_rollout.obs.size()[2:]
            action_shape = pct_rollout.actions.size()[-1]
            leaf_node_value, selectedlogProb, dist_entropy = self.PCT_policy.evaluate_actions(pct_rollout.obs[:-1].view(-1, *obs_shape),
                                                                                            pct_rollout.actions.view(-1, action_shape),
                                                                                            normFactor=factor)
            leaf_node_value = leaf_node_value.view(num_steps, num_processes, 1)
            selectedlogProb = selectedlogProb.view(num_steps, num_processes, 1)

            advantages = pct_rollout.returns[:-1] - leaf_node_value
            critic_loss = advantages.pow(2).mean()
            actor_loss  = -(advantages.detach() * selectedlogProb).mean()

            if self.use_acktr and self.policy_optim.steps % self.policy_optim.Ts == 0:
                # Sampled fisher, see Martens 2014d
                self.PCT_policy.zero_grad()
                pg_fisher_loss = - selectedlogProb.mean()

                value_noise = torch.randn(leaf_node_value.size())
                if leaf_node_value.is_cuda:
                    value_noise = value_noise.to(device)

                sample_values = leaf_node_value + value_noise
                vf_fisher_loss = -(leaf_node_value - sample_values.detach()).pow(2).mean()

                fisher_loss = pg_fisher_loss + vf_fisher_loss
                self.policy_optim.acc_stats = True
                fisher_loss.backward(retain_graph=True)
                self.policy_optim.acc_stats = False

            self.policy_optim.zero_grad()
            (args.actor_loss_coef * actor_loss
             + args.critic_loss_coef  * critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.PCT_policy.parameters(), args.max_grad_norm)
            self.policy_optim.step()

            ##############################################
            ############ After optimzation ###############
            ##############################################
            pct_rollout.after_update()

            # Save the trained policy model
            if (self.step_counter % args.model_save_interval == 0) and args.model_save_path != "":
                if self.step_counter % args.model_update_interval == 0:
                    sub_time_str = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))

                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)

                torch.save(self.PCT_policy.state_dict(), os.path.join(model_save_path, 'PCT-' + self.timeStr + '_' + sub_time_str + ".pt"))

            # Write tensorboard logs.
            if self.step_counter % args.print_log_interval == 0 and len(episode_rewards)>1:
                total_num_steps = (self.step_counter + 1 - inside_counter) * num_processes * num_steps
                end = time.time()
                if len(episode_ratio) != 0:
                    ratio_recorder = max(ratio_recorder, np.max(episode_ratio))

                print("Time version: {} is training\n"
                      "Updates {}, num timesteps {}, FPS {}\n"
                      "Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                      "The dist entropy {:.5f}, the value loss {:.5f}, the action loss {:.5f}\n"
                      "The mean space ratio is {}, the ratio threshold is{}\n"
                        .format(self.timeStr,
                                self.step_counter, total_num_steps, int(total_num_steps / (end - start)),
                                len(episode_rewards), np.mean(episode_rewards), np.median(episode_rewards), np.min(episode_rewards), np.max(episode_rewards),
                                dist_entropy.item(), critic_loss.item(), actor_loss.item(),
                                np.mean(episode_ratio), ratio_recorder))
                self.writer.add_scalar('Rewards/Mean rewards', np.mean(episode_rewards), self.step_counter)
                self.writer.add_scalar("Rewards/Max rewards", np.max(episode_rewards), self.step_counter)
                self.writer.add_scalar('Rewards/Min rewards', np.min(episode_rewards), self.step_counter)
                self.writer.add_scalar("Ratio/The max ratio", np.max(episode_ratio), self.step_counter)
                self.writer.add_scalar("Ratio/The mean ratio", np.mean(episode_ratio), self.step_counter)
                self.writer.add_scalar("Ratio/The max ratio in history", ratio_recorder, self.step_counter)
                self.writer.add_scalar("Training/Value loss", critic_loss.item(), self.step_counter)
                self.writer.add_scalar("Training/Action loss", actor_loss.item(), self.step_counter)
                self.writer.add_scalar('Training/Distribution entropy', dist_entropy.item(), self.step_counter)
