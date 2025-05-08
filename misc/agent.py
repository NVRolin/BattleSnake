import torch
import torch.nn as nn

from misc.train import spaceInvaders

print("Torch version:",torch.__version__)
from tqdm import trange
import torch.optim as optim
import os
import json
from datetime import datetime
import numpy as np
import copy
import replayBuffer
import model
import draw
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/DQN")
class Agent(object):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''

    def __init__(self, n_actions: int):
        self._n_actions = n_actions
        self._last_action = None

    def _forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        pass

    def _backward(self):
        ''' Performs a backward pass on the network '''
        pass

    def _forward_processing(self, env, state, action):
        ''' Performs the processing necessary for a forward pass'''
        pass


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''

    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def _forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.__last_action = np.random.randint(self._n_actions)
        return self.__last_action

    def _forward_processing(self, env, state, action):
        action = self._forward(state)

        next_state, reward, terminal, truncated, _ = env.step(action)
        done = terminal or truncated
        return env, next_state, action, reward, done




class DQNAgent(Agent):
    def __init__(self, discount_factor, buffer_size, neural_network, optimizer, n_actions, n_frames):
        super(DQNAgent, self).__init__(n_actions)

        self.__experience_replay_buffer = replayBuffer.ExperienceReplayBuffer(capacity=buffer_size)  # init the exp replay buffer
        self.__n_frames = n_frames
        self.__steps = 0
        self.__total_steps = 0
        self.__nn = neural_network.cuda()
        if self.__nn.get_device() == "cuda":
            self.__nn = self.__nn.cuda()  # specify the device of the neural network to cuda.
        self.__discount_factor = torch.tensor(discount_factor, device=self.__nn.get_device(), requires_grad=False)
        # Save the params used by the agent for loading and saving.
        self.__params_used = {}
        self.__params_used["buffer_size"] = buffer_size
        self.__params_used["n_frames"] = n_frames
        self.__params_used["discount_factor"] = discount_factor
        self.__params_used["n_actions"] = int(n_actions)
        self.__params_used["input_size"] = int(neural_network.get_input_size())
        self.__params_used["hidden_size"] = int(neural_network.get_hidden_size())
        self.__params_used["output_size"] = int(neural_network.get_output_size())
        self.__params_used["device"] = str(neural_network.get_device())
        self.__params_used["lr"] = float(optimizer.param_groups[0]['lr'])
        self.__loss = nn.SmoothL1Loss()  # We can also use nn.MSELoss()

        # nn.MSELoss()
        # calculate the mean square error from the state action values between our network
        # and the target network and save it to a tensor.
        # nn.SmoothL1Loss
        # Creates a criterion that uses a squared term if the absolute element-wise
        # error falls below beta and an L1 term otherwise.
        # less sensitive to outliers than torch.nn.MSELoss and in some cases prevents exploding gradients
        self.__optimizer = optimizer  # init the optimizer
        # Register hooks to monitor gradients
        self.__gradients = []

        self.__target_nn = copy.deepcopy(self.__nn)  # init target network as copy of main network.
        for p in self.__target_nn.parameters():
            p.requires_grad = False
        if self.__nn.get_device() == "cuda":
            self.__target_nn = self.__target_nn.cuda()  # specify the device of the neural network to cuda.

    # We do not want these actions to be recorded for our next calculation of the gradient.
    """
    @torch.no_grad()
    def __forward_DQN(self, state, epsilon): # forward pass for DQN
        if np.random.random() < epsilon: # random action
            self.__last_action = int(np.random.choice(self.__n_actions,1,replace=False))
        else: # greedy action
            q_values = self.__nn(torch.tensor(state, requires_grad=False,device=self.nn.get_device()))
            action = q_values.max(0)[1]
            self.__last_action = int(action.item())
        return self.__last_action
    """

    @torch.no_grad()
    def _forward(self, state, epsilon):  # forward pass for DQN+CNN
        if np.random.random() < epsilon:  # random action
            self.__last_action = np.random.randint(self._n_actions)
        else:  # greedy action
            # with profiler.profile(with_stack=True, profile_memory=True) as PROF:
            state = state.detach().to(self.__nn.get_device(), dtype=torch.float) / 255
            q_values = self.__nn(state)
            # writer.add_graph(self.__nn, state)
            # writer.close()
            # print(q_values)
            # detatch returns the state tensor without the gradient, so it has no attachments with the current gradients.
            action = torch.argmax(q_values)
            self.__last_action = action.item()

        return self.__last_action

    @torch.no_grad()
    def _forwardBoltz(self, state, tau):
        # boltzman
        state = state.detach().to(self.__nn.get_device(), dtype=torch.float) / 255
        q_values = self.__nn(state).cpu().numpy()[0]
        max_q_value = np.max(q_values)
        exp_prob = np.exp((q_values - max_q_value) / tau)
        prob = exp_prob / np.sum(exp_prob)
        # choose actions according to the probabilities
        action = np.random.choice(range(self._n_actions), p=prob)
        self.__last_action = action
        return self.__last_action

    def _backward(self, N):  # backward pass - based on current batch of samples update weights of agent's nn
        if self.__experience_replay_buffer.get_capacity() < N:  # not enough samples in buffer
            return
        # self.nn.zero_grad()  essentially the same operation as optimizer.zero_grad(), but it's applied directly to the neural network's parameters rather than through the optimizer.
        self.__optimizer.zero_grad()  # We clear out the gradients of all parameters that the optimizer is tracking.
        # Not calling it can lead to incorrect gradient computations since we only want to compute the gradients for a single batch.

        state, action, reward, next_state, done = self.__experience_replay_buffer.sample_batch(
            N)  # take N samples from the buffer
        state = state.to(self.__nn.get_device(), dtype=torch.float) / 255
        action = action.to(self.__nn.get_device(), dtype=torch.int64)
        reward = reward.to(self.__nn.get_device(), dtype=torch.float)
        next_state = next_state.to(self.__nn.get_device(), dtype=torch.float) / 255
        done = done.to(self.__nn.get_device())
        # We want to get the maximum along the second dimension of the tensor, and [0] accesses the value not the indices of the tensor.
        # for i in range(len(reward)):
        #    if done[i]:
        #        print(reward[i])
        #        print(done)
        #        plt.imshow(np.array(state.squeeze(0)[i][0].cpu()))
        #        plt.show()
        #        plt.imshow(np.array(next_state.squeeze(0)[i][0].cpu()))
        #        plt.show()
        with torch.no_grad():  # Detach qvalues_next_state from the computation graph
            qvalues_next_state = self.__target_nn(next_state).max(dim=1)[0]

        # We want to get the state action values from the neural network
        # After passing the state through the neural network, we get a tensor of Q-values for all actions.
        # we want to select the Q-value corresponding to the action that was actually taken.
        # This is done using the gather operation: At the second(1) dimension
        # action.view(N, 1) reshapes the action tensor to have a shape compatible with gathering.
        # N is the batch size so every action in a batch is print(action.view(N, 1))
        state_action_values = self.__nn(state).gather(1, action.view(N, 1))
        # We calculate the TD target using the immediate rewards with the discounted estimate optimal q value for the next state if it terminated.
        target_TD = reward + self.__discount_factor * qvalues_next_state

        target_values = done * reward + (~done) * target_TD

        # state_action_values.view(-1) flattens the tensor to 1 dim
        # We compute the loss with optimizer specified with self.loss
        loss = self.__loss(state_action_values.view(-1), target_values.detach())
        # loss = self.__loss(state_action_values, target_values.unsqueeze(1))
        loss.backward()  # Computes the gradient of the loss tensor.
        ret_loss = loss.cpu().detach()  # We return the loss for plotting purposes
        torch.nn.utils.clip_grad_norm_(self.__nn.parameters(),
                                       1)  # We clip the gradient to avoid the exploding gradient phenomenon.
        self.__optimizer.step()  # we compute the back propagation

        for p in self.__nn.parameters():
            self.__gradients.append(float(p.grad.norm().detach().cpu()))
        return ret_loss

    def __update_target_network(self):
        self.__target_nn.load_state_dict(
            self.__nn.state_dict())  # Load_state_dict is a integral entry that can save or load models from PyTorch, it contains information about the neural networks state.
        # self.__target_nn.eval()

    def __fill_buffer(self, env):
        # function for initial filling of the buffer
        # Note that in order to start training i.e. finding Q values you need a filled buffer
        # but you cannot fill a buffer without Q values (which actions to take?)
        # so instead, you can fill it initially by choosing uniformly random actions
        if spaceInvaders:
            state = env.reset()[0]  # you start new episode by using env.reset, and the function returns initial state
        else:
            env.reset()
            state = env.get_observation()
            state = state["board"]
        state = self.__process(state)  # Preprocess the input
        done = False
        reward = 0
        action_for_frames = 0
        stacked_state, reward, done,_ = self.__stack_frames(env, state, action_for_frames, reward,
                                                          done)  # stack first frames
        print("Filling buffer...")
        for i in range(self.__experience_replay_buffer.get_capacity()):
            # Current state frame processing
            # We compute the action for the current state.
            action_for_frames = np.random.randint(self._n_actions)
            # plt.imshow(np.array(state.squeeze(0)[0].cpu()))
            # plt.show()
            # We stack the frame with the action for the current state.
            # The method will append frames for the first frames in a episode
            # Next state
            if spaceInvaders:
                next_state, reward, terminal, truncated, _ = env.step(action_for_frames)
                done = terminal or truncated
            else:
                next_state, reward, terminal, _ = env.step([action_for_frames])
                next_state = next_state["board"]
                reward = reward[0]
                done = terminal
            if done:
                stacked_next_state = stacked_state
            else:
                # next state processing
                next_state = self.__process(next_state)
                stacked_next_state, reward, done, _= self.__stack_frames(env, next_state, action_for_frames, reward, done)
                # if not ended, next state becomes current
            reward_tensor = torch.tensor([reward], dtype=torch.uint8, requires_grad=False)
            action_for_frames_tensor = torch.tensor([action_for_frames], dtype=torch.uint8, requires_grad=False)
            done_tensor = torch.tensor([done], dtype=torch.bool, requires_grad=False)

            # Form Experience tuple from state, action, reward, next_state, done, and append it to the buffer
            exp = replayBuffer.Experience(stacked_state.detach(), action_for_frames_tensor.detach(), reward_tensor.detach(),
                                          stacked_next_state.detach(), done_tensor.detach())
            self.__experience_replay_buffer.append(exp)
            if done:
                if spaceInvaders:
                    state = env.reset()[0]
                else:
                    env.reset()
                    state = env.get_observation()
                    state = state["board"]
                state = self.__process(state)
                done = False
                reward = 0
                action_for_frames = 0
                stacked_state, reward, done, _ = self.__stack_frames(env, state, action_for_frames, reward, done)
            else:
                stacked_state = stacked_next_state
        print("Buffer filled!")
    def __process(self, state):
        # state = rescale(state, scale=0.5)
        if spaceInvaders:
            state = state[np.newaxis, np.newaxis, :, :]

            state = torch.tensor(np.array(state), dtype=torch.uint8, requires_grad=False)

            # state_tensor2 = state_tensor2[..., 0]
            # state_tensor2 = state_tensor2.unsqueeze(0)
            # print(state_tensor2.size())
        else:
            state = state[np.newaxis, np.newaxis, :, :]
            state = torch.tensor(np.array(state), dtype=torch.uint8, requires_grad=False)
            # state_tensor2 = state_tensor2[..., 0]
            # state_tensor2 = state_tensor2.unsqueeze(0)
            # print(state_tensor2.size())
        return state

    def __stack_frames(self, env, current_state, current_action, tot_reward, done,info ={}):
        # wrapper_args_no_frameskip = self.__params_used['wrapper_args'].copy()  # Create a copy to avoid modifying the original dictionary
        # wrapper_args_no_frameskip['frame_skip'] = 1  # Update frame_skip value
        # If we are done at the start of a stack stack the done frame
        while current_state.size()[1] < self.__n_frames:
            if not done:
                # new_frame, reward, terminal, truncated, _ = gym.wrappers.AtariPreprocessing(env.unwrapped, **wrapper_args_no_frameskip).step(current_action)
                if spaceInvaders:
                    new_frame, reward, terminal, truncated, info = env.step(current_action)
                else:
                    new_frame, reward, terminal, info = env.step([current_action])
                    reward = reward[0]
                    new_frame = new_frame['board']
                self.__steps += 1
                self.__total_steps += 1
                tot_reward += reward
                if spaceInvaders:
                    done = terminal or truncated
                else:
                    done = terminal
                if done:
                    new_frame = np.array(current_state.squeeze(0)[0].cpu())
                new_frame_tensor = self.__process(new_frame)
                current_state = torch.cat([current_state, new_frame_tensor], 1)
            else:
                # plt.imshow(new_frame)
                # plt.show()
                current_state = torch.cat([current_state, new_frame_tensor], 1)
        return current_state, tot_reward, done,info

    def __running_average(self, x, N):
        # Function used to compute the running average of the last N elements of a vector x
        # if x shorter than N, return zeros. Use np.convolve to find averages
        if len(x) < N:
            y = np.zeros_like(x)
        else:
            # y = np.convolve(x, np.ones(N)/N,mode='valid') # valid seems to have the same effect as len(x) < N ?
            y = np.copy(x)
            y[N - 1:] = np.convolve(x, np.ones((N,)) / N, mode='valid')
        return y


    def train_policy(self, hyperparams, env, wrapper_args):
        """
        Train the DQN agent using either episode-based or step-based training.

        Args:
            hyperparams: Dictionary containing hyperparameters
            env: Training environment
            wrapper_args: Arguments for the environment wrapper
        """
        # Store hyperparameters in agent's params
        self.__store_hyperparameters(hyperparams, env, wrapper_args)

        # Extract hyperparameters
        n_steps = hyperparams['n_steps']
        n_episodes = hyperparams['n_episodes']
        n_ep_running_average = hyperparams['n_ep_running_average']
        target_network_update_freq = hyperparams['target_network_update_freq']
        batch_size = hyperparams['batch_size']
        eps_min = hyperparams['eps_min']
        eps_max = hyperparams['eps_max']

        # Episode-based training
        if n_episodes != 0 and n_steps == 0:
            self.__train_with_episodes(n_episodes, n_ep_running_average, target_network_update_freq,
                                       batch_size, eps_min, eps_max, env)

        # Step-based training
        elif n_episodes == 0 and n_steps != 0:
            self.__train_with_steps(n_steps, n_ep_running_average, target_network_update_freq,
                                    batch_size, eps_min, eps_max, env)

        # Invalid configuration
        else:
            return

    def __store_hyperparameters(self, hyperparams, env, wrapper_args):
        """Store the hyperparameters used for training in the agent's parameter dictionary."""
        self.__params_used['n_episodes'] = hyperparams['n_episodes']
        self.__params_used['n_ep_running_average'] = hyperparams['n_ep_running_average']
        self.__params_used['target_network_update_freq'] = hyperparams['target_network_update_freq']
        self.__params_used['batch_size'] = hyperparams['batch_size']
        self.__params_used['eps_min'] = hyperparams['eps_min']
        self.__params_used['eps_max'] = hyperparams['eps_max']
        self.__params_used['wrapper_args'] = wrapper_args
        if spaceInvaders:
            self.__params_used['env_name'] = env.spec.id
        else:
            self.__params_used['env_name'] = env.NAME

    def __train_with_episodes(self, n_episodes, n_ep_running_average, target_network_update_freq,
                              batch_size, eps_min, eps_max, env):
        """Train the agent using episode-based training."""
        # Setup epsilon decay
        epsilon_decay = int(0.99 * n_episodes)
        epsilon_factor = (eps_min / eps_max) ** (1 / epsilon_decay)

        # Initialize training data storage
        self.__initialize_training_data_lists()

        # Fill the replay buffer initially
        self.__fill_buffer(env)

        # Record start time and initialize step counter
        self.__params_used['start_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.__total_steps = 0

        # Training loop
        episodes = trange(n_episodes, desc='Episode: ', leave=True)
        for i in episodes:
            # Get epsilon for this episode
            epsilon = max(eps_min, eps_max * (epsilon_factor ** i))

            # Run a single episode and collect data
            episode_stats = self.__run_episode(env, epsilon, target_network_update_freq, batch_size)

            # Record episode results
            self.__record_episode_stats(episode_stats, epsilon, n_ep_running_average)

            # Update progress bar description
            self.__update_progress_description(episodes, i, episode_stats['reward_avg'], epsilon)

        # Cleanup
        self.__cleanup(env)

    def __train_with_steps(self, n_steps, n_ep_running_average, target_network_update_freq,
                           batch_size, eps_min, eps_max, env):
        """Train the agent using step-based training."""
        # Setup epsilon decay
        epsilon_decay = int(0.99 * n_steps)
        epsilon_factor = (eps_min / eps_max) ** (1 / epsilon_decay)

        # Initialize training data storage
        self.__initialize_training_data_lists()

        # Fill the replay buffer initially
        self.__fill_buffer(env)

        # Record start time and initialize step counter
        self.__params_used['start_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.__total_steps = 0

        # Training loop
        steps = trange(n_steps, desc='Step: ', leave=True)
        i = 0

        while i < len(steps):
            # Get epsilon based on current step
            epsilon = max(eps_min, eps_max * (epsilon_factor ** i))

            # Run a single episode
            episode_stats = self.__run_episode(env, epsilon, target_network_update_freq, batch_size)
            i = self.__total_steps  # Update step counter

            # Record episode results
            self.__record_episode_stats(episode_stats, epsilon, n_ep_running_average)

            # Update progress bar with step information
            ep = len(self.__ep_reward_list)
            self.__update_steps_progress_description(steps, i, ep, episode_stats['reward_avg'], epsilon)

        # Cleanup
        self.__cleanup(env)

    def __initialize_training_data_lists(self):
        """Initialize lists to track training data."""
        self.__ep_reward_list = []  # Used to save episodes reward
        self.__ep_reward_list_RA = []  # Used to save the computed running average
        self.__ep_steps_list = []  # Used to save number of steps per episode
        self.__eps_list = []  # Used to save the epsilon for each episode
        self.__loss_list = []  # Used to keep track of the loss over the training
        self.__mean_grad = []  # Used to track mean gradients

    def __run_episode(self, env, epsilon, target_network_update_freq, batch_size):
        """Run a single training episode and return stats."""
        # Reset environment
        if spaceInvaders:
            state = env.reset()[0]
        else:
            state = env.reset()
            state = state["board"]
        state = self.__process(state)

        # Initialize episode variables
        done = False
        next_action = 0
        total_episode_reward = 0
        reward = 0
        self.__steps = 0
        loss = None

        # Stack initial frames
        stacked_next_state, reward, done,_ = self.__stack_frames(env, state, next_action, reward, done)

        # Episode loop
        while not done:
            # Set current state
            stacked_state = stacked_next_state

            # Choose action using epsilon-greedy policy
            next_action = self._forward(stacked_state, epsilon)
            # Take action in environment
            if done:  # Will never happen
                stacked_next_state = stacked_state
            else:
                # Get next state and reward. What are terminal and truncated? If episode ended.
                if spaceInvaders:
                    next_state, reward, terminal_next_state, truncated_next_state, _ = env.step(next_action)
                    done = terminal_next_state or truncated_next_state
                else:
                    next_state, reward, terminal_next_state, _ = env.step([next_action])
                    next_state = next_state["board"]
                    reward = reward[0]
                    done = terminal_next_state
                if done:
                    stacked_next_state = stacked_state
                else:
                    next_state = self.__process(next_state)
                    stacked_next_state, reward, done,_ = self.__stack_frames(env, next_state, next_action, reward,
                                                                           done)
            # Update counters and rewards
            total_episode_reward += reward
            self.__steps += 1
            self.__total_steps += 1

            # Store experience in replay buffer
            self.__store_experience(stacked_state, next_action, reward, stacked_next_state, done)

            # Update neural network
            loss = self._backward(batch_size)

            # Update target network periodically
            if self.__total_steps % target_network_update_freq == 0:
                self.__update_target_network()

        # Return episode statistics
        return {
            'reward': total_episode_reward,
            'steps': self.__steps,
            'loss': float(loss) if loss is not None else 0,
            'reward_avg': self.__running_average(self.__ep_reward_list + [total_episode_reward],
                                                 len(self.__ep_reward_list) + 1)[-1]
        }

    def __store_experience(self, state, action, reward, next_state, done):
        """Store an experience tuple in the replay buffer."""
        reward_tensor = torch.tensor([reward], dtype=torch.uint8, requires_grad=False)
        action_tensor = torch.tensor([action], dtype=torch.uint8, requires_grad=False)
        done_tensor = torch.tensor([done], dtype=torch.bool, requires_grad=False)

        exp = replayBuffer.Experience(state.detach(), action_tensor.detach(),
                                      reward_tensor.detach(), next_state.detach(),
                                      done_tensor.detach())
        self.__experience_replay_buffer.append(exp)

    def __record_episode_stats(self, stats, epsilon, n_ep_running_average):
        """Record episode statistics for tracking and visualization."""
        self.__ep_reward_list.append(stats['reward'])
        current_ra = self.__running_average(self.__ep_reward_list, n_ep_running_average)[-1]
        self.__ep_reward_list_RA.append(current_ra)
        self.__ep_steps_list.append(stats['steps'])
        self.__eps_list.append(epsilon)
        self.__loss_list.append(stats['loss'])
        self.__mean_grad.append(np.mean(self.__gradients))

        # Log metrics to TensorBoard
        writer.add_scalar('training loss', stats['loss'], self.__total_steps)
        writer.add_scalar('Episode reward', stats['reward'], self.__total_steps)

        # Reset gradients for next episode
        self.__gradients = []

    def __update_progress_description(self, progress_bar, iteration, reward_avg, epsilon):
        """Update the progress bar description with current training stats."""
        progress_bar.set_description(
            "Episode {} - Reward/epsilon: {:.1f}/{:.2f} - Avg. # of steps: {}| Max grad: {:.5E} | Min grad: {:.5E} | Mean grad: {:.5E}".format(
                iteration, reward_avg, epsilon,
                self.__running_average(self.__ep_steps_list, 10)[-1],
                max(self.__gradients) if self.__gradients else 0,
                min(self.__gradients) if self.__gradients else 0,
                np.mean(self.__gradients) if self.__gradients else 0
            )
        )

    def __update_steps_progress_description(self, progress_bar, step, episode, reward_avg, epsilon):
        """Update the steps progress bar description with current training stats."""
        progress_bar.set_description(
            "Step {:.3E} Episode {} - Reward/epsilon: {:.1f}/{:.2f} - Avg. # of steps: {}| Max grad: {:.5E} | Min grad: {:.5E} | Mean grad: {:.5E}".format(
                step, episode, reward_avg, epsilon,
                self.__running_average(self.__ep_steps_list, 10)[-1],
                max(self.__gradients) if self.__gradients else 0,
                min(self.__gradients) if self.__gradients else 0,
                np.mean(self.__gradients) if self.__gradients else 0
            )
        )

    def __cleanup(self, env):
        """Perform cleanup operations after training."""
        writer.close()
        self.__params_used['total_steps'] = self.__total_steps
        self.__experience_replay_buffer.clear_memory()
        if spaceInvaders:
            env.close()



    def _forward_processing(self, env, state, action, reward, done,info={}):
        info_next_state = {}
        if not spaceInvaders:
            state = state["board"]
        if not torch.is_tensor(state):
            state = self.__process(state)
        stacked_state, reward, done,info = self.__stack_frames(env, state, action, reward, done,info)
        action = self._forward(stacked_state, 0)
        if done:
            stacked_next_state = stacked_state
            reward_next_state = reward
            done_next_state = done
        else:
            # Get next state and reward. What are terminal and truncated? If episode ended.
            if spaceInvaders:
                next_state, reward_next_state, terminal_next_state, truncated_next_state, info_next_state = env.step(action)
                done_next_state = terminal_next_state or truncated_next_state
            else:
                next_state, reward_next_state, terminal_next_state, info_next_state = env.step([action])
                next_state = next_state["board"]
                reward_next_state = reward_next_state[0]
                done_next_state = terminal_next_state
            if done_next_state:
                stacked_next_state = stacked_state
                reward_next_state = reward
                done_next_state = done_next_state
            else:
                next_state = self.__process(next_state)
                stacked_next_state, reward_next_state, done_next_state,info_next_state = self.__stack_frames(env, next_state, action,
                                                                                             reward_next_state,
                                                                                             done_next_state,info_next_state)
                reward_next_state += reward
        # Append the next state to the stack of frames.
        # stacked_next_state = torch.cat([stacked_state, next_state], 1)
        # Remove the oldest frame so that we have a correct frame count.
        # stacked_next_state = stacked_next_state[:, 1:, :, :]
        return env, stacked_next_state, action, reward_next_state, done_next_state,info_next_state

    def test_policy(self, env1, n_episodes=50, dual=True, env2=None):
        if dual is True:
            print('Init random agent...')
            random_agent = RandomAgent(env2.action_space.n)
            Rewards = []
            # Set up the rendering window
        else:
            print('Solo test agent...')
        # Simulate episodes
        print('Checking solution...')
        Rewards = []
        try:
            for i in range(n_episodes):
                if spaceInvaders:
                    draw_env = draw.DrawEnvironment(env1, dual, env2=env2)
                else:
                    draw_env = draw.DrawBattlesnakeEnvironment(env1, dual, env2=env2)
                if dual is True:
                    Rewards.append(draw_env.run(self, random_agent)[0][0])
                else:
                    Rewards.append(draw_env.run(self)[0][0])
                draw_env.reset()
            draw_env.close()
        except(KeyboardInterrupt):
            print("Keyboard Interrupt!")
            print("Closing the tests...")
            draw_env.close()
            return None
        return Rewards

    def __save_model(self, path):  # We save the model along with its arguments, buffer is saved separately
        torch.save({
            'neural_network_state_dict': self.__nn.state_dict(),
            'optimizer_state_dict': self.__optimizer.state_dict(),
        }, path)

    def __load_model(self, path):  # We load the model from the path we saved to.
        checkpoint = torch.load(path)
        self.__nn.load_state_dict(checkpoint['neural_network_state_dict'])
        self.__optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def get_ep_reward_list(self):
        return self.__ep_reward_list

    def get_ep_reward_list_RA(self):
        return self.__ep_reward_list_RA

    def get_ep_steps_list(self):
        return self.__ep_steps_list

    def get_eps_list(self):
        return self.__eps_list

    def get_loss_list(self):
        return self.__loss_list

    def __make_numerated_dir_path(self, save_dir):
        # Create the save directory
        dir_name = os.path.basename(save_dir)
        # Extract the directory path
        parent_dir = os.path.dirname(save_dir)

        # Create a numerated dict structure
        i = 0
        while True:
            new_dir_name = f"{dir_name}_{i}"
            new_save_dir = os.path.join(parent_dir, new_dir_name)
            if not os.path.exists(new_save_dir):
                break
            i += 1
        save_dir = new_save_dir
        return save_dir

    def save_model_and_parameters(self, save_dir, desc):
        # Make numerated dir for the test
        save_dir = self.__make_numerated_dir_path(save_dir)

        os.makedirs(save_dir)

        self.__params_used['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Save date
        self.__params_used['desc'] = desc

        # Save parameters used for training
        with open(os.path.join(save_dir, 'parameters.json'), 'w') as f:
            json.dump(self.__params_used, f, indent=4)

        # Save trained model
        model_path = os.path.join(save_dir, 'model.pt')
        self.__save_model(model_path)

        # Save additional training data if there is any
        try:
            with open(os.path.join(save_dir, 'training_data.json'), 'w') as f:
                json.dump({
                    'ep_reward_list': self.__ep_reward_list,
                    'ep_reward_list_RA': self.__ep_reward_list_RA,
                    'ep_steps_list': self.__ep_steps_list,
                    'loss_list': self.__loss_list,
                    'eps_list': self.__eps_list,
                    'mean_grad': self.__mean_grad
                }, f, indent=4)
        except:
            print("Could not find train data.")
        print("Saved!")

    @classmethod
    def load_models_and_parameters_DQN_CNN(self, dir_path, env):
        if not os.path.isdir(dir_path):
            raise Exception("Invalid path for model")

        # Load parameters from the 'parameters.json' file
        params_file = os.path.join(dir_path, 'parameters.json')
        with open(params_file, 'r') as f:
            self.__params_used = json.load(f)
        print(self.__params_used)
        # Init the model from params
        n_actions = env.action_space.n  # Number of actions

        # Ensure the neural network and the data are on the same device
        print("Trying to use device:", self.__params_used['device'])
        if self.__params_used['device'] == "cuda":  # test i we can run cuda, if not we use cpu
            print("Is CUDA enabled?", torch.cuda.is_available())
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        main_network = model.DQNetworkCNN(self.__params_used['output_size'], self.__params_used['input_size'],
                                       self.__params_used['hidden_size'], device)
        main_network = main_network.to(main_network.get_device())  # Move main network to Device
        print("Using: " + str(main_network.get_device()))

        optimizer = optim.RMSprop(main_network.parameters(), lr=self.__params_used['lr'])
        # Define DQN agent
        dqn_agent = DQNAgent(self.__params_used['discount_factor'], self.__params_used['buffer_size'], main_network,
                             optimizer, n_actions, self.__params_used['n_frames'])

        # Load the trained model from the 'model.pt' file
        model_path = os.path.join(dir_path, 'model.pt')
        dqn_agent.__load_model(model_path)
        # Assign the params
        dqn_agent.__params_used = self.__params_used
        # Load additional training data
        try:
            training_data_file = os.path.join(dir_path, 'training_data.json')
            with open(training_data_file, 'r') as f:
                training_data = json.load(f)
            # Assign loaded training data
            dqn_agent.__ep_reward_list = training_data['ep_reward_list']
            dqn_agent.__ep_reward_list_RA = training_data['ep_reward_list_RA']
            dqn_agent.__ep_steps_list = training_data['ep_steps_list']
            dqn_agent.__loss_list = training_data['loss_list']
            dqn_agent.__eps_list = training_data['eps_list']
            dqn_agent.__mean_grad = training_data['mean_grad']
        except:
            print("Could not find train data.")
        print("Loaded!")
        return dqn_agent