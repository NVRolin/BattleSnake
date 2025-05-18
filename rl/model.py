import torch.nn as nn
import torch
import torch.nn.init as init
class DQNetworkCNNNEW(nn.Module):
    # Create a feedforward neural network. Class contains two functions:
    # init is used for setting architecture of the neural network
    # forward is used for calculating output of the nn given an input
    def __init__(self, output_size, input_size, hidden_size, device):
        super(DQNetworkCNNNEW, self).__init__()  # inherrent values from parrent class.
        self.__device = device  # cuda or cpu
        self.__input_size = input_size
        self.__output_size = output_size
        self.__hidden_size = hidden_size

        self.__c1_block = nn.Sequential(
            nn.Conv2d(self.__input_size, 32, kernel_size=2),
            nn.LeakyReLU(),

            nn.Conv2d(32, 64, kernel_size=2),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.LeakyReLU()
        )
        FC_size = self._get_conv_out(self.__input_size)

        self.__FC_block = nn.Sequential(
            # nn.Flatten(), #Flatten into single vector
            nn.Linear(in_features=FC_size, out_features=self.__hidden_size),
            nn.LeakyReLU(),
            # nn.Linear(in_features=self.__hidden_size,  out_features=self.__hidden_size),
            # nn.ReLU(),
            nn.Linear(in_features=self.__hidden_size, out_features=self.__output_size)
        )

        # Initialize linear layers with kaiming initialization
        for layer in self.__FC_block:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, nonlinearity="leaky_relu")
        # Initialize conv2d layers with kaiming initialization
        for layer in self.__c1_block:
            if isinstance(layer, nn.Conv2d):
                init.kaiming_uniform_(layer.weight, nonlinearity="leaky_relu")
        print(self.get_num_parameters())
    def forward(self, x):
        # with profiler.record_function("LINEAR PASS"):
        x = self.__c1_block(x)
        # with profiler.record_function("Conv PASS"):
        x = self.__FC_block(x.view(x.size(0), -1))
        return x

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def _get_conv_out(self, in_channels):
        # run a dummy 11×11 through conv block to get flatten size
        x = torch.zeros(1, in_channels, 11, 11)
        x = self.__c1_block(x)
        return int(x.numel())
    def get_device(self):
        return self.__device

    def get_input_size(self):
        return self.__input_size

    def get_output_size(self):
        return self.__output_size

    def get_hidden_size(self):
        return self.__hidden_size

class DQNetworkCNN(nn.Module):
    # Create a feedforward neural network. Class contains two functions:
    # init is used for setting architecture of the neural network
    # forward is used for calculating output of the nn given an input
    def __init__(self, output_size, input_size, hidden_size, device):
        super(DQNetworkCNN, self).__init__()  # inherrent values from parrent class.
        self.__device = device  # cuda or cpu
        self.__input_size = input_size
        self.__output_size = output_size
        self.__hidden_size = hidden_size

        self.__c1_block = nn.Sequential(
            nn.Conv2d(self.__input_size, 32, kernel_size=3),
            nn.LeakyReLU(),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.LeakyReLU()
        )
        FC_size = self._get_conv_out(self.__input_size)

        self.__FC_block = nn.Sequential(
            # nn.Flatten(), #Flatten into single vector
            nn.Linear(in_features=FC_size, out_features=self.__hidden_size),
            nn.LeakyReLU(),
            # nn.Linear(in_features=self.__hidden_size,  out_features=self.__hidden_size),
            # nn.ReLU(),
            nn.Linear(in_features=self.__hidden_size, out_features=self.__output_size)
        )

        # Initialize linear layers with kaiming initialization
        for layer in self.__FC_block:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, nonlinearity="leaky_relu")
        # Initialize conv2d layers with kaiming initialization
        for layer in self.__c1_block:
            if isinstance(layer, nn.Conv2d):
                init.kaiming_uniform_(layer.weight, nonlinearity="leaky_relu")
        print(self.get_num_parameters())
    def forward(self, x):
        # with profiler.record_function("LINEAR PASS"):
        x = self.__c1_block(x)
        # with profiler.record_function("Conv PASS"):
        x = self.__FC_block(x.view(x.size(0), -1))
        return x

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def _get_conv_out(self, in_channels):
        # run a dummy 11×11 through conv block to get flatten size
        x = torch.zeros(1, in_channels, 11, 11)
        x = self.__c1_block(x)
        return int(x.numel())
    def get_device(self):
        return self.__device

    def get_input_size(self):
        return self.__input_size

    def get_output_size(self):
        return self.__output_size

    def get_hidden_size(self):
        return self.__hidden_size

class DQNetwork(nn.Module):
    def __init__(self, n_actions, n_inputs,device):
        super().__init__()
        hidden = 64
        self.device = device
        self.layer1 = nn.Linear(n_inputs, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.layer3 = nn.Linear(hidden, hidden)
        self.adv = nn.Linear(hidden, n_actions)

        self.l1_activation = nn.ReLU()
        self.l2_activation = nn.ReLU()
        self.l3_activation = nn.ReLU()

    def forward(self, x):
        y1 = self.l1_activation(self.layer1(x))
        y1 = self.l2_activation(self.layer2(y1))
        y1 = self.l3_activation(self.layer3(y1))
        adv = self.adv(y1)
        return adv
