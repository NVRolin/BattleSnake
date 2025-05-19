import numpy as np
import torch
import torch.optim as optim
from rl.model import *
from rl.agent import *
from gym.env import *

if __name__ == "__main__":
    device = torch.device('cuda')
    path = "./models/experiments/"
    seed = None
    np.random.seed(seed)

    env = BattlesnakeEnv(11, 4)
    env_name = env.NAME
    n_actions = len(env.ACTIONS)

    base_params = {
        'n_episodes': 2000,
        'n_steps': 0,
        'input_size': 13,
        'buffer_size': 10000,
        'hidden_size': 512,
        'output_size': 4,
        'device': "cuda",
        'discount_factor': 0.99,
        'n_ep_running_average': 10,
        'alpha': 0.001,
        'target_network_update_freq': 50,
        'batch_size': 256,
        'eps_min': 0.1,
        'eps_max': 1,
        'n_frames': 13,
        'times_tested': 1,
        'friendly_model': "./models/experiments/_7",
        'enemy_model': "./models/experiments/_7",
        'env': env_name,
        'seed': seed
    }

    # ensure the neural network and the data are on the same device
    if base_params['device'] == "cuda":
        print("Is CUDA enabled?", torch.cuda.is_available())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    main_network = DQNModel(n_actions, base_params['n_frames'], base_params['hidden_size'], device)
    main_network = main_network.to(device)

    # init the optimizer
    optimizer = optim.RMSprop(main_network.parameters(), lr=base_params['alpha'])

    # train the agent
    dqn_agent = DQNAgent(base_params['discount_factor'], base_params['buffer_size'], main_network, optimizer, n_actions, base_params['n_frames'])
    dqn_agent.train(base_params, env)
    dqn_agent.save(path)