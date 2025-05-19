import copy
import json
import os
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from rl.model import *
from rl.buffer import *
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/DQN")
print("Torch version:",torch.__version__)

class DQNAgent():
    def __init__(self, discount_factor, buffer_size, neural_network, optimizer, n_actions, n_frames):
        super(DQNAgent, self).__init__()
        device = neural_network.get_device()
        self._experience_replay_buffer = ExperienceReplayBuffer(capacity=buffer_size)
        self._steps = 0
        self._total_steps = 0
        self._nn = neural_network.to(device)
        self._discount_factor = discount_factor
        self._params_used = {
            "buffer_size": buffer_size,
            "n_frames": n_frames,
            "discount_factor": discount_factor,
            "n_actions": int(n_actions),
            "device": str(device),
            "lr": float(optimizer.param_groups[0]['lr']),
            "input_size": neural_network.get_input_size(),
            "hidden_size": neural_network.get_hidden_size(),
            "output_size": neural_network.get_output_size()
        }
        self._loss = nn.SmoothL1Loss()
        self.dqn_agent_friend = None
        self.dqn_agent_enemy_one = None
        self.dqn_agent_enemy_two = None
        self._head_positions = [(0,0),(0,0),(0,0),(0,0)]

        self._optimizer = optimizer
        self._gradients = []

        # target network does not require grad
        self._target_nn = copy.deepcopy(self._nn) 
        for p in self._target_nn.parameters():
            p.requires_grad = False

        if self._nn.get_device() == "cuda":
            self._target_nn = self._target_nn.cuda() 


    @torch.no_grad()
    def _forward(self, state, epsilon, head_positions, state_tensor=None):
        # forward pass based on epsilon
        if np.random.random() < epsilon:  # random action
            actions = [0, 1, 2, 3]
            random.shuffle(actions)
            candidates_tensor = torch.tensor(actions, dtype=torch.int64, requires_grad=False, device=self._nn.get_device())
            self.choose_action(candidates_tensor, state / 255.0, head_positions, True)
        else:  # greedy action
            state_tensor = state_tensor.to(dtype=torch.float32).div_(255.0).unsqueeze(0)
            candidates_tensor = self._nn(state_tensor).squeeze(0)
            self.choose_action(candidates_tensor, state / 255.0, head_positions)
        return self._last_action
    

    def _backward(self): # backward pass based on current batch of samples
        batch_size = self._params_used['batch_size']
        if len(self._experience_replay_buffer) < batch_size:
            return # not enough samples in buffer
        self._optimizer.zero_grad() # clear out the gradients of all parameters that the optimizer is tracking between batches
        state_uint8, action, reward, next_state_uint8, done,indices,weights = self._experience_replay_buffer.sample_batch(batch_size)  # take N samples from the buffer
        state = state_uint8.to(torch.float32).div_(255.0)
        next_state = next_state_uint8.to(torch.float32).div_(255.0)

        q = self._nn(state) # [N, A]
        # we want to get the state action values from the neural network
        # after passing the state through the neural network, we get a tensor of q-values for all actions
        # we want to select the q-value corresponding to the action that was actually taken.
        state_action_values = q.gather(1, action.view(batch_size, 1)).view(-1) # [N]
        next_q = self._target_nn(next_state).max(1)[0] # [N]
        # We calculate the tD target using the immediate rewards with the discounted estimate optimal q-value for the next state if it terminated
        td_target = reward + self._discount_factor * next_q * (~done)

        loss = self._loss(state_action_values, td_target.detach())
        loss.backward() # computes the gradient of the loss tensor
        torch.nn.utils.clip_grad_norm_(self._nn.parameters(), 1) # clip the gradient to avoid the exploding gradients
        self._optimizer.step() # backprop

        # compute new priorities
        with torch.no_grad():
            new_errors = (state_action_values - td_target).abs().cpu().numpy()
        # update the sampled transitions' priorities
        self._experience_replay_buffer.update_priorities(indices, new_errors)
        for p in self._nn.parameters():
            self._gradients.append(float(p.grad.norm().detach().cpu()))
        return loss.detach().cpu()


    def _fill_buffer(self, env):
        # in order to start training, we need to fill the buffer
        # the buffer needs to be filled with experiences of the form (state, action, reward, next_state, done)
        # bootstrap-dilemma: we need experiences to learn q-values, but we need q-values to get experiences
        env.reset()
        action_for_frames = []
        # get current state
        state, rot_state = self._stack_frames(env)
        print("Filling buffer...")

        for _ in trange(self._experience_replay_buffer.get_capacity(), desc="Buffer", ncols=150, unit='exp'):
            # current state frame processing
            action_for_frames = []
            agentAction = self._forward(state[0], 1, self._head_positions[0])
            action_for_frames.append(agentAction)

            # get actions for the agents
            # if the agent is not none use the agent's action, otherwise use the random action
            if self.dqn_agent_friend is None:
                action_for_frames.append(self._forward(state[1], 1, self._head_positions[1]))
            else:
                action_for_frames.append(self.dqn_agent_friend._forward(state[1], 1, self._head_positions[1]))

            if self.dqn_agent_enemy_one is None:
                action_for_frames.append(self._forward(state[2], 1, self._head_positions[2]))
            else:
                action_for_frames.append(self.dqn_agent_enemy_one._forward(state[2], 1, self._head_positions[2]))

            if self.dqn_agent_enemy_two is None:
                action_for_frames.append(self._forward(state[3], 1, self._head_positions[3]))
            else:
                action_for_frames.append(self.dqn_agent_enemy_two._forward(state[3], 1, self._head_positions[3]))
                
            _, reward, done, _ = env.step(action_for_frames)
            reward = reward[0]

            # next state processing
            next_state, rot_next_state = self._stack_frames(env)
            # store the experience in the buffer
            s1 = torch.unsqueeze(torch.tensor(rot_state[0], dtype=torch.uint8, device=self._nn.get_device()), 0)
            a = torch.tensor([agentAction], dtype=torch.int64, device=self._nn.get_device())
            r = torch.tensor([reward], dtype=torch.float32, device=self._nn.get_device())
            s2 = torch.unsqueeze(torch.tensor(rot_next_state[0], dtype=torch.uint8, device=self._nn.get_device()), 0)
            d = torch.tensor([done], dtype=torch.bool, device=self._nn.get_device())

            exp = Experience(s1, a, r, s2, d)
            self._experience_replay_buffer.append(exp)

            # we check if the environment is done
            if done:
                # reset the environment and begin another episode
                env.reset()
                state, rot_state = self._stack_frames(env)
            else:
                # set the current state to the next state
                state = next_state
                rot_state = rot_next_state

        print("Buffer filled!")

    def _stack_frames(self, env):
        # battlesnake stacking of frames inspired by https://medium.com/asymptoticlabs/battlesnake-post-mortem-a5917f9a3428
        obs = env.get_observation()
        current_states = []
        rot_states = []
        B = env.board_size

        for snake_idx in range(env.n_snakes):
            health_frame = np.zeros((B, B), dtype=np.uint8) # health at head
            bin_body_frame = np.zeros((B, B), dtype=np.uint8) # snake body with values 255
            segment_body_frame = np.zeros((B, B), dtype=np.uint8) # snake body with increasing segment length
            longer_opponent_frame = np.zeros((B, B), dtype=np.uint8) # longer opponent head with value 255
            food_frame = np.zeros((B, B), dtype=np.uint8) # food positions with value 255
            board_frame = np.full((B, B), 255, dtype=np.uint8) # board with value 255
            agent_head_frame = np.zeros((B, B), dtype=np.uint8) # agent heads with value 255
            double_tail_frame = np.zeros((B, B), dtype=np.uint8) # double tail with value 255
            longer_size_frame = np.zeros((B, B), dtype=np.uint8) # longer opponent snake body with value 255
            shorter_size_frame = np.zeros((B, B), dtype=np.uint8) # shorter opponent snake body with value 255
            alive_count_frames = np.zeros((3, B, B), dtype=np.uint8) # alive count frames with value 255
            
            if snake_idx == 0:
                if obs['alive'][1]:
                    alive_count_frames[0].fill(255)
                if obs['alive'][2]:
                    alive_count_frames[1].fill(255)
                if obs['alive'][3]:
                    alive_count_frames[2].fill(255)
            elif snake_idx == 1:
                if obs['alive'][0]:
                    alive_count_frames[0].fill(255)
                if obs['alive'][2]:
                    alive_count_frames[1].fill(255)
                if obs['alive'][3]:
                    alive_count_frames[2].fill(255)
            elif snake_idx == 2:
                if obs['alive'][3]:
                    alive_count_frames[0].fill(255)
                if obs['alive'][0]:
                    alive_count_frames[1].fill(255)
                if obs['alive'][1]:
                    alive_count_frames[2].fill(255)
            elif snake_idx == 3:
                if obs['alive'][2]:
                    alive_count_frames[0].fill(255)
                if obs['alive'][0]:
                    alive_count_frames[1].fill(255)
                if obs['alive'][1]:
                    alive_count_frames[2].fill(255)

            for i in range(len(obs['snakes'])):
                if not obs['alive'][i]:
                    continue
                snake = obs['snakes'][i]
                head_x, head_y = snake[0]
                health = obs['health'][i]
                health_frame[head_y, head_x] = health * 255 // 100
                for j in range(len(snake)):
                    x, y = snake[j]
                    bin_body_frame[y, x] = 255
                    segment_body_frame[y, x] = j
                    if len(snake) >= len(obs['snakes'][snake_idx]):
                        longer_size_frame[y, x] = len(snake)-len(obs['snakes'][snake_idx])
                    elif len(snake) < len(obs['snakes'][snake_idx]):
                        shorter_size_frame[y, x] = len(obs['snakes'][snake_idx])-len(snake)
                if len(snake) >= len(obs['snakes'][snake_idx]):
                    longer_opponent_frame[head_y, head_x] = 255
                if obs['food_eaten'][i]:
                    double_tail_x, double_tail_y = snake[-1]
                    double_tail_frame[double_tail_y, double_tail_x] = 255
            for x, y in obs['food']:
                food_frame[y, x] = 255
            head_x, head_y = obs['snakes'][snake_idx][0]

            agent_head_frame[head_y, head_x] = 255

            self._head_positions[snake_idx] = (head_x, head_y)
            all_frames = np.stack([
                health_frame,
                bin_body_frame,
                segment_body_frame,
                longer_opponent_frame,
                food_frame,
                board_frame,
                agent_head_frame,
                double_tail_frame,
                longer_size_frame,
                shorter_size_frame,
                *alive_count_frames
            ], axis=0)

            current_states.append(all_frames)
            rot_states.append(all_frames.copy())
            
        return np.stack(current_states, axis=0), np.stack(rot_states, axis=0)

    def choose_action(self, candidates, state, head_position, random=False):
        # choose the action based on the candidates and the state
        # we avoid immediate death
        head_x, head_y = head_position
        # define possible moves: up, down, left, right
        H, W = state.shape[1], state.shape[2]
        move_cells = [
            (max(head_y - 1, 0), head_x),
            (min(head_y + 1, H - 1), head_x),
            (head_y, max(head_x - 1, 0)),
            (head_y, min(head_x + 1, W - 1))
        ]
        # extract relevant state layers to avoid immediate death
        body_layer = state[1]
        larger_opponent_layer = state[3]

        # init mask lists
        blocked_actions = [head_y == 0, head_y == H - 1, head_x == 0, head_x == W - 1]
        risky_actions = [False] * 4

        # check collisions
        for i, (y, x) in enumerate(move_cells):
            if body_layer[y, x] == 1:
                blocked_actions[i] = True
            # check neighbors for larger opponents
            for ny, nx in [(max(y - 1, 0), x), (min(y + 1, H - 1), x), (y, max(x - 1, 0)), (y, min(x + 1, W - 1))]:
                if larger_opponent_layer[ny, nx] == 1:
                    risky_actions[i] = True
                    break

        # convert boolean masks to torch tensors
        blocked_mask = torch.tensor(blocked_actions, dtype=torch.bool, device=self._nn.get_device())
        risky_mask = torch.tensor(risky_actions, dtype=torch.bool, device=self._nn.get_device())

        if not random:
            # mask out blocked actions
            valid_vals = candidates.clone()
            valid_vals[blocked_mask] = float('-inf')

            # if all moves blocked, pick best unmasked
            if torch.all(valid_vals == float('-inf')):
                action = torch.argmax(candidates).item()
            else:
                # further mask risky
                final_vals = valid_vals.clone()
                final_vals[risky_mask] = float('-inf')
                if torch.all(final_vals == float('-inf')):
                    action = torch.argmax(valid_vals).item()
                else:
                    action = torch.argmax(final_vals).item()
            self._last_action = action
        else:
            # select candidates that are not blocked and not risky
            valid_mask = (~blocked_mask[candidates]) & (~risky_mask[candidates])
            valid_candidates = candidates[valid_mask]

            if valid_candidates.numel() > 0:
                c_action = valid_candidates[0]
            else:
                # not blocked
                unblocked_mask = ~blocked_mask[candidates]
                unblocked_candidates = candidates[unblocked_mask]

                if unblocked_candidates.numel() > 0:
                    c_action = unblocked_candidates[0]
                else:
                    # all blocked
                    c_action = candidates[0]

            self._last_action = c_action.item()


    def _running_average(self, x):
        # running average of the last elements of a vector x
        N = 30
        if len(x) < N:
            y = np.zeros_like(x)
        else:
            y = np.copy(x)
            y[N - 1:] = np.convolve(x, np.ones((N,)) / N, mode='valid')
        return y[-1]


    def train(self, hyperparams, env):
        # store the hyperparameters
        self._store_hyperparameters(hyperparams, env)

        self.dqn_agent_friend = load_dqn_agent(hyperparams['friendly_model'], env, old_model=True)
        self.dqn_agent_enemy_one = load_dqn_agent(hyperparams['enemy_model'], env, old_model=True)
        self.dqn_agent_enemy_two = load_dqn_agent(hyperparams['enemy_model'], env, old_model=True)
        
        # epsilon decay
        epsilon_factor = (self._params_used['eps_min'] / self._params_used['eps_max']) ** (1 / int(0.99 * self._params_used['n_episodes']))

        # initialize training data storage
        self._ep_reward_list = [] # episodes reward
        self._ep_reward_list_ra = [] # computed running average
        self._ep_steps_list = [] # number of steps per episode
        self._eps_list = [] # epsilon for each episode
        self._loss_list = [] # loss over the training
        self._mean_grad = [] # mean gradients

        # fill the replay buffer initially
        self._fill_buffer(env)

        # record start time and init step counter
        self._params_used['start_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._total_steps = 0

        # training loop
        episodes = trange(self._params_used['n_episodes'], desc='Episode: ', leave=True, ncols=150, unit='ep')
        for i in episodes:
            # run a single episode
            epsilon = max(self._params_used['eps_min'], self._params_used['eps_max'] * (epsilon_factor ** i))
            episode_stats = self._run_episode(env, epsilon, self._params_used['target_network_update_freq'], self._params_used['batch_size'])

            # record episode results
            self._record_episode_stats(episode_stats, epsilon)

            # Update progress bar description
            episodes.set_description(
            "Episode {} - R: {:.1f} - eps: {:.2f} - avg steps: {} - max grad: {:.2E} - min grad: {:.2E}".format(
                i, episode_stats['reward_avg'], epsilon,
                self._running_average(self._ep_steps_list),
                max(self._gradients) if self._gradients else 0,
                min(self._gradients) if self._gradients else 0,
                )
            )
            self._gradients = []

        writer.close()
        self._params_used['total_steps'] = self._total_steps


    def _store_hyperparameters(self, hyperparams, env):
        """Store the hyperparameters used for training in the agent's parameter dictionary."""
        self._params_used['n_episodes'] = hyperparams['n_episodes']
        self._params_used['n_ep_running_average'] = hyperparams['n_ep_running_average']
        self._params_used['target_network_update_freq'] = hyperparams['target_network_update_freq']
        self._params_used['batch_size'] = hyperparams['batch_size']
        self._params_used['eps_min'] = hyperparams['eps_min']
        self._params_used['eps_max'] = hyperparams['eps_max']
        self._params_used['times_tested'] = hyperparams['times_tested']
        self._params_used['friendly_model'] = hyperparams['friendly_model']
        self._params_used['env_name'] = env.NAME


    def _run_episode(self, env, epsilon, target_network_update_freq, batch_size):
        env.reset()

        # init variables
        done = False
        next_action = []
        total_episode_reward = 0
        reward = 0
        self._steps = 0
        loss = None
        steps_since_target = 0

        # stack initial frames
        state, rot_state = self._stack_frames(env)
        state_tensor = torch.tensor(rot_state, dtype=torch.uint8, requires_grad=False, device=self._nn.get_device())

        # episode loop
        while not done:
            next_action = []
            # choose action using epsilon-greedy policy
            agentAction = self._forward(state[0], epsilon, self._head_positions[0],state_tensor[0])
            next_action.append(agentAction)

            # get actions for the agents
            if self.dqn_agent_friend is None:
                next_action.append(self._forward(state[1], 0, self._head_positions[1],state_tensor[1]))
            else:
                next_action.append(self.dqn_agent_friend._forward(state[1], 0, self._head_positions[1],state_tensor[1]))

            if self.dqn_agent_enemy_one is None:
                next_action.append(self._forward(state[2], 0, self._head_positions[2],state_tensor[2]))
            else:
                next_action.append(self.dqn_agent_enemy_one._forward(state[2], 0, self._head_positions[2],state_tensor[2]))

            if self.dqn_agent_enemy_two is None:
                next_action.append(self._forward(state[3], 0, self._head_positions[3],state_tensor[3]))
            else:
                next_action.append(self.dqn_agent_enemy_two._forward(state[3], 0, self._head_positions[3],state_tensor[3]))
            
            # process next state
            _, rewards, done, _ = env.step(next_action)
            reward = rewards[0]  # Get the reward for our agent
            next_state, rot_next_state = self._stack_frames(env)
            next_state_tensor = torch.tensor(rot_next_state, dtype=torch.uint8, requires_grad=False, device=self._nn.get_device())

            # update counters and rewards
            total_episode_reward += reward
            self._steps += 1
            self._total_steps += 1
            steps_since_target +=1

            # store experience in replay buffer
            exp = Experience(
                state_tensor[0].unsqueeze(0),
                torch.tensor([agentAction], dtype=torch.int64, device=self._nn.get_device()),
                torch.tensor([reward], dtype=torch.float32, device=self._nn.get_device()),
                next_state_tensor[0].unsqueeze(0),
                torch.tensor([done], dtype=torch.bool, device=self._nn.get_device()),
            )

            # store with no error
            self._experience_replay_buffer.append(exp)
            loss = self._backward()

            # update target network periodically
            if steps_since_target >= target_network_update_freq == 0:
                self._target_nn.load_state_dict(
                    self._nn.state_dict()) 
                steps_since_target = 0
            state = next_state
            state_tensor = next_state_tensor

        return {
            'reward': total_episode_reward,
            'steps': self._steps,
            'loss': float(loss) if loss is not None else 0,
            'reward_avg': self._running_average(self._ep_reward_list + [total_episode_reward])
        }


    def _record_episode_stats(self, stats, epsilon):
        # record episode results
        self._ep_reward_list.append(stats['reward'])
        self._ep_reward_list_ra.append(self._running_average(self._ep_reward_list))
        self._ep_steps_list.append(stats['steps'])
        self._eps_list.append(epsilon)
        self._loss_list.append(stats['loss'])
        self._mean_grad.append(np.mean(self._gradients))
        
        # log metrics to tensorboard
        writer.add_scalar('Training loss', stats['loss'], self._total_steps)
        writer.add_scalar('Episode reward', stats['reward'], self._total_steps)


    def save(self, save_dir):
        dir_name = os.path.basename(save_dir)
        parent_dir = os.path.dirname(save_dir)

        i = 0
        while True:
            new_dir_name = f"{dir_name}_{i:02d}"
            new_save_dir = os.path.join(parent_dir, new_dir_name)
            if not os.path.exists(new_save_dir):
                break
            i += 1
        save_dir = new_save_dir
        os.makedirs(save_dir)

        self._params_used['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Save date

        # save the parameters and the model
        with open(os.path.join(save_dir, 'parameters.json'), 'w') as f:
            json.dump(self._params_used, f, indent=4)

        torch.save({
            'neural_network_state_dict': self._nn.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
        }, os.path.join(save_dir, 'model.pt'))

        # save additional training data
        with open(os.path.join(save_dir, 'training_data.json'), 'w') as f:
            json.dump({
            'ep_reward_list': self._ep_reward_list,
            'ep_reward_list_ra': self._ep_reward_list_ra,
            'ep_steps_list': self._ep_steps_list,
            'loss_list': self._loss_list,
            'eps_list': self._eps_list,
            'mean_grad': self._mean_grad
            }, f, indent=4)

        print("Saved!")


def load_dqn_agent(dir_path, env, old_model=False):

    if not os.path.isdir(dir_path):
        raise Exception(f"Invalid path for model: {dir_path}")

    # load parameters
    params_file = os.path.join(dir_path, 'parameters.json')
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    # Get action space size from environment
    n_actions = len(env.ACTIONS)
    
    if params['device'] == "cuda":
        print("Is CUDA enabled?", torch.cuda.is_available())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print("Using device:", device) 

    # create the network with the saved parameters
    if old_model:
        main_network = DQNetworkCNN(params['output_size'], params['input_size'], params['hidden_size'], device)
    else:
        main_network = DQNModel(params['output_size'], params['input_size'], params['hidden_size'], device)
    main_network = main_network.to(device)

    # init the optimizer and agent
    optimizer = optim.RMSprop(main_network.parameters(), lr=params['lr'])
    dqn_agent = DQNAgent(params['discount_factor'], params['buffer_size'], main_network, optimizer, n_actions, params['n_frames'])

    # load the saved model weights
    checkpoint = torch.load(os.path.join(dir_path, 'model.pt'), map_location=device)
    dqn_agent._nn.load_state_dict(checkpoint['neural_network_state_dict'])
    dqn_agent._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Ensure model is on the correct device
    dqn_agent._nn = dqn_agent._nn.to(device)
    dqn_agent._target_nn = dqn_agent._target_nn.to(device)

    # set the parameters
    dqn_agent._params_used = params

    # sync the target network
    dqn_agent._target_nn.load_state_dict(dqn_agent._nn.state_dict())

    # load the training data
    try:
        with open(os.path.join(dir_path, 'training_data.json'), 'r') as f:
                training_data = json.load(f)

        dqn_agent._ep_reward_list = training_data['ep_reward_list']
        dqn_agent._ep_reward_list_ra = training_data['ep_reward_list_ra']
        dqn_agent._ep_steps_list = training_data['ep_steps_list']
        dqn_agent._loss_list = training_data['loss_list']
        dqn_agent._eps_list = training_data['eps_list']
        dqn_agent._mean_grad = training_data['mean_grad']

    except Exception as e:
        print(f"Error loading training data: {e}")

    print("Model loaded successfully!")
    return dqn_agent
