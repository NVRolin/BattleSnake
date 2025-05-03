import numpy as np
import torch
import gymnasium as gym
import torch.optim as optim
import ale_py
import NN
import Agent
device = torch.device('cuda')

if __name__ == "__main__":

    do_train = True

    path = "./models/SpaceInvaders/"
    seed = None
    np.random.seed(seed)
    wrapper_args = {
        "terminal_on_life_loss": False,
        "frame_skip": 4,
        "grayscale_obs": True,
        "scale_obs": False,
    }
    if do_train is True:
        env_name = "ALE/SpaceInvaders-v5"
        env = gym.make(env_name, render_mode="rgb_array", frameskip=1)
        env = gym.wrappers.AtariPreprocessing(env, **wrapper_args)
        env.action_space.seed(seed)
        n_actions = env.action_space.n  # Number of actions

        desc = "Basic save and load test"

        base_params = {
            'n_episodes': 20,
            'n_steps': 0,
            'buffer_size': 2200,
            'hidden_size': 512,
            'device': "cuda",
            'discount_factor': 0.99,
            'n_ep_running_average': 10,
            'alpha': 0.0001,
            'target_network_update_freq': 500,
            'batch_size': 32,
            'eps_min': 0.1,
            'eps_max': 1,
            'n_frames': 4,
            'times_tested': 1,  # The number to tests we want to execute with these parameters
            'env': env_name,
            'seed': seed
        }

        # Ensure the neural network and the data are on the same device
        if base_params['device'] == "cuda":  # test i we can run cuda, if not we use cpu
            print("Is CUDA enabled?", torch.cuda.is_available())
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        main_network = NN.DQNetworkCNN(n_actions, base_params['n_frames'], base_params['hidden_size'], device)
        if main_network.get_device() == "cuda":
            main_network = main_network.to(
                main_network.get_device())  # Move main network to Device for cuda to work, (Better if we do this in the mynetwork class TODO)
        print("Using: " + str(main_network.get_device()))

        # Init the optimizer
        # optimizer = optim.Adam(main_network.parameters(), lr=base_params['alpha'])
        optimizer = optim.RMSprop(main_network.parameters(), lr=base_params['alpha'])
        # Define DQN agent
        dqn_agent = Agent.DQNAgent(base_params['discount_factor'], base_params['buffer_size'], main_network, optimizer,
                                   n_actions, base_params['n_frames'])
        dqn_agent.train_policy(base_params, env, wrapper_args)
        dqn_agent.save_model_and_parameters(save_dir=path, desc=desc)
        env_name = "ALE/SpaceInvaders-v5"

        env1 = gym.make(env_name, render_mode="human", frameskip=1, repeat_action_probability=0)
        env1 = gym.wrappers.AtariPreprocessing(env1, **wrapper_args)
        env1.metadata['render_fps'] = 60

        # env2 = gym.make(env_name, render_mode="human")
        # env2 = stable_baselines3.common.atari_wrappers.AtariWrapper(env2)
        # env2.metadata['render_fps'] =60

        dqn_agent.test_policy(env1,2,dual=False)
        # env2 = env2.close()
        env1 = env1.close()