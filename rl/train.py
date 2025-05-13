import numpy as np
import torch
import torch.optim as optim
import model
import agent
import gym.env
device = torch.device('cuda')
if __name__ == "__main__":

    do_train = True

    path = "./models/experiments/"
    seed = None
    np.random.seed(seed)
    if do_train is True:
        env = gym.env.BattlesnakeEnv(11,4)
        env_name = env.NAME
        n_actions = len(env.ACTIONS)
        desc = "Basic save and load test"

        base_params = {
            'n_episodes': 20000,
            'n_steps': 0,
            'buffer_size': 20000,
            'hidden_size': 512,
            'device': "cuda",
            'discount_factor': 0.999,
            'n_ep_running_average': 10,
            'alpha': 0.001,
            'target_network_update_freq': 50,
            'batch_size': 32,
            'eps_min': 0.1,
            'eps_max': 1,
            'n_frames': 13,
            'times_tested': 1,
            'friendly_model': "./models/experiments/_11",
            'enemy_model': "./models/experiments/_11",
            'env': env_name,
            'seed': seed
        }

        # Ensure the neural network and the data are on the same device
        if base_params['device'] == "cuda":  # test i we can run cuda, if not we use cpu
            print("Is CUDA enabled?", torch.cuda.is_available())
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        main_network = model.DQNetworkCNN(n_actions, base_params['n_frames'], base_params['hidden_size'], device)
        if main_network.get_device() == "cuda":
            main_network = main_network.to(
                main_network.get_device())  # Move main network to Device for cuda to work, (Better if we do this in the mynetwork class TODO)
        print("Using: " + str(main_network.get_device()))

        # Init the optimizer
        # optimizer = optim.Adam(main_network.parameters(), lr=base_params['alpha'])
        optimizer = optim.RMSprop(main_network.parameters(), lr=base_params['alpha'])
        # Define DQN agent
        dqn_agent = agent.DQNAgent(base_params['discount_factor'], base_params['buffer_size'], main_network, optimizer,
                                   n_actions, base_params['n_frames'])
        dqn_agent.train_policy(base_params, env)
        dqn_agent.save_model_and_parameters(save_dir=path, desc=desc)
        env = gym.env.BattlesnakeEnv(11,4)

        dqn_agent.test_policy(env,20)
    else:
        # Load the model and parameters
        path = "./models/experiments/_11"
        dual = True
        if dual is True:
            env = gym.env.BattlesnakeEnv(11,4)
        else:
            env = gym.env.BattlesnakeEnv(11,1)
        dqn_agent = agent.DQNAgent.load_models_and_parameters_DQN_CNN(path, env)
        if dual:
            dqn_agent.dqn_agent_friend = agent.DQNAgent.load_models_and_parameters_DQN_CNN(path, env)
            dqn_agent.dqn_agent_enemy1 = agent.DQNAgent.load_models_and_parameters_DQN_CNN(path, env)
            dqn_agent.dqn_agent_enemy2 = agent.DQNAgent.load_models_and_parameters_DQN_CNN(path, env)
        dqn_agent.test_policy(env,20)
